import argparse
import functools
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import cupy as cp
from gluonts.dataset.arrow import ArrowWriter
from sklearn.gaussian_process.kernels import (
    ExpSineSquared,
    DotProduct,
    RBF,
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel,
)

# -----------------------------------------------------------------------------
# 1. Kernel Bank Construction (parameterised by `time_length`)
# -----------------------------------------------------------------------------

def build_kernel_bank(time_length: int) -> List:
    """Return a list of base kernels whose characteristic *periodicity* is
    expressed as a *fraction* of the desired time‑series length.  Because the
    fractional ratio is preserved, changing ``time_length`` automatically
    rescales every kernel.  This eliminates the hidden dependency on a global
    constant and fully "unplugs" the hyper‑parameter at the heart of this
    refactor.

    Parameters
    ----------
    time_length : int
        Length of the (discrete) time axis along which each series is sampled.
    """

    return [
        # ---- Hourly / sub‑hourly cycles -------------------------------------------------
        ExpSineSquared(periodicity=24 / time_length),  # 1 hour (if 1 unit ≙ 1 min)
        ExpSineSquared(periodicity=48 / time_length),  # 30 minutes
        ExpSineSquared(periodicity=96 / time_length),  # 15 minutes
        # ---- Hourly components embedded in weekly structure ---------------------------
        ExpSineSquared(periodicity=24 * 7 / time_length),   # 1 h within a week
        ExpSineSquared(periodicity=48 * 7 / time_length),   # 30 min within a week
        ExpSineSquared(periodicity=96 * 7 / time_length),   # 15 min within a week
        # ---- Daily / sub‑daily --------------------------------------------------------
        ExpSineSquared(periodicity=7 / time_length),   # 1 day
        ExpSineSquared(periodicity=14 / time_length),  # 12 hours
        ExpSineSquared(periodicity=30 / time_length),  # 1 month (≈30 days)
        ExpSineSquared(periodicity=60 / time_length),  # 15 days
        ExpSineSquared(periodicity=365 / time_length), # 1 year
        ExpSineSquared(periodicity=365 * 2 / time_length), # 6 months
        # ---- Weekly / monthly / quarterly variations ----------------------------------
        ExpSineSquared(periodicity=4 / time_length),   # 1 week
        ExpSineSquared(periodicity=26 / time_length),  # half‑year in weeks
        ExpSineSquared(periodicity=52 / time_length),  # 1 year in weeks
        ExpSineSquared(periodicity=4 / time_length),   # 1 month again (redundant but
                                                      # kept to preserve original list)
        ExpSineSquared(periodicity=6 / time_length),   # 2 months
        ExpSineSquared(periodicity=12 / time_length),  # 3 months / quarter
        ExpSineSquared(periodicity=4 / time_length),   # 1 quarter again (redundant)
        ExpSineSquared(periodicity=(4 * 10) / time_length), # 10 quarters
        ExpSineSquared(periodicity=10 / time_length),  # 10 days
        # ---- Stationary + noise kernels ----------------------------------------------
        DotProduct(sigma_0=0.0),
        DotProduct(sigma_0=1.0),
        DotProduct(sigma_0=10.0),
        RBF(length_scale=0.1),
        RBF(length_scale=1.0),
        RBF(length_scale=10.0),
        RationalQuadratic(alpha=0.1),
        RationalQuadratic(alpha=1.0),
        RationalQuadratic(alpha=10.0),
        WhiteKernel(noise_level=0.1),
        WhiteKernel(noise_level=1.0),
        ConstantKernel(),
    ]


# -----------------------------------------------------------------------------
# 2. Binary map utility for kernel algebra
# -----------------------------------------------------------------------------

def random_binary_map(a, b):
    """Randomly combine two kernels via *addition* or *product*."""

    binary_ops = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_ops)(a, b)


# -----------------------------------------------------------------------------
# 3. Mean‑function library (unchanged)
# -----------------------------------------------------------------------------

def zero_mean(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def linear_mean(x: np.ndarray) -> np.ndarray:
    a = np.random.uniform(-1.0, 1.0)
    b = np.random.uniform(-1.0, 1.0)
    return a * x + b


def exponential_mean(x: np.ndarray) -> np.ndarray:
    a = np.random.uniform(0.5, 1.5)
    b = np.random.uniform(0.5, 1.5)
    return a * np.exp(b * x)


def anomaly_mean(x: np.ndarray) -> np.ndarray:
    m = np.zeros_like(x)
    num_anomalies = np.random.randint(1, 6)
    for _ in range(num_anomalies):
        idx = np.random.randint(0, len(x))
        m[idx] += np.random.uniform(-5.0, 5.0)
    return m


def random_mean_combination(x: np.ndarray) -> np.ndarray:
    """Pick two base mean functions and combine them at random (\+ or \*)."""

    mean_functions = [zero_mean, linear_mean, exponential_mean, anomaly_mean]
    m1, m2 = np.random.choice(mean_functions, 2, replace=True)
    combine_ops = [lambda u, v: u + v, lambda u, v: u * v]
    return np.random.choice(combine_ops)(m1(x), m2(x))


# -----------------------------------------------------------------------------
# 4. GPU‑accelerated sampling from the GP prior (unchanged)
# -----------------------------------------------------------------------------

def sample_from_gp_prior_efficient_gpu(
    kernel,
    X: np.ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
    mean_vec: Optional[np.ndarray] = None,
):
    """Draw one realisation from the GP prior *on the GPU*.

    A user‑provided ``mean_vec`` enables arbitrary mean‑functions.  The input
    array ``X`` is expected to be one‑dimensional so we reshape it into a
    design‑matrix of shape *(n, 1)* for kernel evaluation.
    """

    if X.ndim == 1:
        X = X[:, None]

    cov_cpu = kernel(X)
    n = X.shape[0]

    # Default to a zero‑vector on GPU if no explicit mean was given.
    mean_vec = np.zeros(n, dtype=np.float64) if mean_vec is None else mean_vec

    cov_gpu = cp.asarray(cov_cpu)
    mean_gpu = cp.asarray(mean_vec)

    if random_seed is not None:
        cp.random.seed(random_seed)

    ts_gpu = cp.random.multivariate_normal(mean=mean_gpu, cov=cov_gpu, method=method)

    return cp.asnumpy(ts_gpu)


# -----------------------------------------------------------------------------
# 5. Structural‑Causal‑Model time‑series generator (parameterised)
# -----------------------------------------------------------------------------

def generate_random_dag(num_nodes: int, max_parents: int = 3) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    G.add_nodes_from(nodes)
    for i in range(num_nodes):
        possible_parents = nodes[:i]
        num_par = np.random.randint(0, min(len(possible_parents), max_parents) + 1)
        for p in random.sample(possible_parents, num_par):
            G.add_edge(p, nodes[i])
    return G


def random_activation(x: np.ndarray, func_type: str = "linear") -> np.ndarray:
    """Apply the chosen non‑linearity to the parent mixture."""

    if func_type == "linear":
        a = np.random.uniform(0.5, 2.0)
        b = np.random.uniform(-1.0, 1.0)
        return a * x + b
    if func_type == "relu":
        return np.maximum(0.0, x)
    if func_type == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if func_type == "sin":
        return np.sin(x)
    if func_type == "mod":
        c = np.random.uniform(1.0, 5.0)
        return np.mod(x, c)
    # default: leaky‑ReLU
    alpha = np.random.uniform(0.01, 0.3)
    return np.where(x > 0, x, alpha * x)


def random_edge_mapping(parents_data: List[np.ndarray]) -> np.ndarray:
    """Combine parent time‑series linearly, then push through a random activation."""

    combined = np.stack(parents_data, axis=1)
    W = np.random.randn(len(parents_data))
    b = np.random.randn()
    non_linear_input = combined @ W + b
    chosen_func = np.random.choice(["linear", "relu", "sigmoid", "sin", "mod", "leakyrelu"])
    return random_activation(non_linear_input, chosen_func)


# -----------------------------------------------------------------------------
# 6. End‑to‑end SCM sampler
# -----------------------------------------------------------------------------

def generate_scm_time_series(
    *,
    time_length: int = 512,
    num_features: int = 6,
    max_parents: int = 3,
    seed: int = 42,
    post_process: bool = True,  # kept for API compatibility (unused)
    num_nodes: int = 6,
) -> Dict[int, np.ndarray]:
    """Generate one *network* of inter‑linked GP‑based time‑series."""

    np.random.seed(seed)
    random.seed(seed)

    dag = generate_random_dag(num_nodes, max_parents=max_parents)
    kernel_bank = build_kernel_bank(time_length)

    root_nodes = [n for n in dag.nodes if dag.in_degree(n) == 0]
    node_data: Dict[int, np.ndarray] = {}

    X = np.linspace(0.0, 1.0, time_length)

    # ---- Sample roots directly from the GP prior ------------------------------------
    for r in root_nodes:
        selected_kernels = np.random.choice(
            kernel_bank, np.random.randint(1, 8), replace=True
        )
        kernel = functools.reduce(random_binary_map, selected_kernels)
        mean_vec = random_mean_combination(X)
        node_data[r] = sample_from_gp_prior_efficient_gpu(
            kernel=kernel, X=X, mean_vec=mean_vec
        )

    # ---- Propagate through DAG ------------------------------------------------------
    for node in nx.topological_sort(dag):
        if node in root_nodes:
            continue
        parents = list(dag.predecessors(node))
        parents_ts = [node_data[p] for p in parents]
        node_data[node] = random_edge_mapping(parents_ts)

    return node_data


# -----------------------------------------------------------------------------
# 7. Convenience wrappers for dataset assembly (parameterised)
# -----------------------------------------------------------------------------

def sample_time_series_from_scm(
    *,
    time_length: int = 512,
    max_parents: int = 3,
    num_features: int = 6,
    num_nodes: int = 6,
    seed: int = 42,
) -> List[Dict]:
    node_data = generate_scm_time_series(
        time_length=time_length,
        max_parents=max_parents,
        num_features=num_features,
        num_nodes=num_nodes,
        seed=seed,
    )

    chosen_nodes = random.sample(list(node_data.keys()), num_features)

    return [
        {
            "start": np.datetime64("2000-01-01 00:00", "s"),
            "target": node_data[n].astype(np.float32),
        }
        for n in chosen_nodes
    ]


def generate_multiple_scm_time_series(
    *,
    num_series: int = 50_000,
    time_length: int = 512,
    num_features: int = 6,
    max_parents: int = 3,
    num_nodes: int = 6,
    seed: int = 42,
) -> List[Dict]:
    """Generate *num_series* individual time‑series by batching SCM draws."""

    results: List[Dict] = []
    base_seed = seed
    for i in range(num_series // num_features):
        current_seed = base_seed + i * 100
        batch = sample_time_series_from_scm(
            time_length=time_length,
            num_features=num_features,
            max_parents=max_parents,
            num_nodes=num_nodes,
            seed=current_seed,
        )
        results.extend(batch)
        if i % 5_000 == 0:
            print(f"[+] Completed {i * num_features} / {num_series} series …")
    return results


# -----------------------------------------------------------------------------
# 8. CLI entry‑point -----------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic SCM‑GP dataset generator")
    parser.add_argument("-N", "--num-series", type=int, default=200000, help="total number of series to generate")
    parser.add_argument("-L", "--time-length", type=int, default=512, help="length of each individual series")
    parser.add_argument("-O", "--output-file", type=str, default="CauKer200K_Kernel7_Parent6.arrow", help="output file name")
    parser.add_argument("-F", "--features", type=int, default=4, help="how many nodes to sample per SCM draw")
    parser.add_argument("-P", "--max-parents", type=int, default=6, help="maximum #parents per node in the DAG")
    parser.add_argument("-M", "--num-nodes", type=int, default=18, help="total nodes in the DAG")

    args = parser.parse_args()

    rng_series = args.num_series
    time_length = args.time_length
    out_path = Path(args.output_file)

    results = generate_multiple_scm_time_series(
        num_series=rng_series,
        time_length=time_length,
        max_parents=args.max_parents,
        num_features=args.features,
        num_nodes=args.num_nodes,
        seed=42,
    )

    ArrowWriter(compression="lz4").write_to_file(results, path=out_path)
    print(f"Arrow dataset saved to {out_path.resolve()}")
