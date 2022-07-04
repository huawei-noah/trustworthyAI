"""
codes for generating synthetic datasets
"""
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import PolynomialFeatures


def generate_W(d=6, prob=0.5, low=0.5, high=2.0):
    """
    generate a random weighted adjaceecy matrix
    :param d: number of nodes
    :param prob: prob of existing an edge
    :return:
    """
    g_random = np.float32(np.random.rand(d,d)<prob)
    g_random = np.tril(g_random, -1)
    U = np.round(np.random.uniform(low=low, high=high, size=[d, d]), 1)
    U[np.random.randn(d, d) < 0] *= -1
    W = (g_random != 0).astype(float) * U
    return W


def gen_data_given_model(b, s, c, n_samples=10, noise_type='lingam', permutate=False):
    """Generate artificial data based on the given model.
       Based on ICA-LiNGAM codes.
       https://github.com/cdt15/lingam
    Parameters
    ----------
    b : numpy.ndarray, shape=(n_features, n_features)
        Strictly lower triangular coefficient matrix.
        NOTE: Each row of `b` corresponds to each variable, i.e., X = BX.
    s : numpy.ndarray, shape=(n_features,)
        Scales of disturbance variables.
    c : numpy.ndarray, shape=(n_features,)
        Means of observed variables.

    Returns
    -------
    xs, b_, c_ : Tuple
        `xs` is observation matrix, where `xs.shape==(n_samples, n_features)`.
        `b_` is permuted coefficient matrix. Note that rows of `b_` correspond
        to columns of `xs`. `c_` if permuted mean vectors.
    """

    n_vars = b.shape[0]

    # Check args
    assert (b.shape == (n_vars, n_vars))
    assert (s.shape == (n_vars,))
    assert (np.sum(np.abs(np.diag(b))) == 0)
    np.allclose(b, np.tril(b))

    if noise_type == 'lingam':
        # Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
        # (<1 gives subgaussian, >1 gives supergaussian)
        q = np.random.rand(n_vars) * 1.1 + 0.5
        ixs = np.where(q > 0.8)
        q[ixs] = q[ixs] + 0.4

        # Generates disturbance variables
        ss = np.random.randn(n_samples, n_vars)
        ss = np.sign(ss) * (np.abs(ss) ** q)

        # Normalizes the disturbance variables to have the appropriate scales
        ss = ss / np.std(ss, axis=0) * s

    elif noise_type == 'gaussian':
        ss = np.random.randn(n_samples, n_vars) * s

    # Generate the data one component at a time
    xs = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        # NOTE: columns of xs and ss correspond to rows of b
        xs[:, i] = ss[:, i] + xs.dot(b[i, :]) + c[i]

        # Permute variables
    b_ = deepcopy(b)
    c_ = deepcopy(c)
    if permutate:
        p = np.random.permutation(n_vars)
        xs[:, :] = xs[:, p]
        b_[:, :] = b_[p, :]
        b_[:, :] = b_[:, p]
        c_[:] = c[p]

    return xs, b_, c_


def gen_data_given_model_2nd_order(b, s, c, n_samples=10, noise_type='lingam', permutate=False):
    """Generate artificial data based on the given model.
       Quadratic functions

    Parameters
    ----------
    b : numpy.ndarray, shape=(n_features, n_features)
        Strictly lower triangular coefficient matrix.
        NOTE: Each row of `b` corresponds to each variable, i.e., X = BX.
    s : numpy.ndarray, shape=(n_features,)
        Scales of disturbance variables.
    c : numpy.ndarray, shape=(n_features,)
        Means of observed variables.

    Returns
    -------
    xs, b_, c_ : Tuple
        `xs` is observation matrix, where `xs.shape==(n_samples, n_features)`.
        `b_` is permuted coefficient matrix. Note that rows of `b_` correspond
        to columns of `xs`. `c_` if permuted mean vectors.

    """
    # rng = np.random.RandomState(random_state)
    n_vars = b.shape[0]

    # Check args
    assert (b.shape == (n_vars, n_vars))
    assert (s.shape == (n_vars,))
    assert (np.sum(np.abs(np.diag(b))) == 0)
    np.allclose(b, np.tril(b))

    if noise_type == 'lingam':
        # Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
        # (<1 gives subgaussian, >1 gives supergaussian)
        q = np.random.rand(n_vars) * 1.1 + 0.5
        ixs = np.where(q > 0.8)
        q[ixs] = q[ixs] + 0.4

        # Generates disturbance variables
        ss = np.random.randn(n_samples, n_vars)
        ss = np.sign(ss) * (np.abs(ss) ** q)

        # Normalizes the disturbance variables to have the appropriate scales
        ss = ss / np.std(ss, axis=0) * s

    elif noise_type == 'gaussian':

        ss = np.random.randn(n_samples, n_vars) * s
    # Generate the data one component at a time

    xs = np.zeros((n_samples, n_vars))
    poly = PolynomialFeatures()
    newb = []

    for i in range(n_vars):
        # NOTE: columns of xs and ss correspond to rows of b
        xs[:, i] = ss[:, i] + c[i]
        col = b[i]
        col_false_true = np.abs(col) > 0.3
        len_parents = int(np.sum(col_false_true))
        if len_parents == 0:
            newb.append(np.zeros(n_vars, ))
            continue
        else:
            X_parents = xs[:, col_false_true]
            X_2nd = poly.fit_transform(X_parents)
            X_2nd = X_2nd[:, 1:]
            dd = X_2nd.shape[1]
            U = np.round(np.random.uniform(low=0.5, high=1.5, size=[dd, ]), 1)
            U[np.random.randn(dd, ) < 0] *= -1
            U[np.random.randn(dd, ) < 0] *= 0
            X_sum = np.sum(U * X_2nd, axis=1)
            xs[:, i] = xs[:, i] + X_sum

            # remove zero-weight variables
            X_train_expand_names = poly.get_feature_names()[1:]
            cj = 0
            new_reg_coeff = np.zeros(n_vars, )

            # hard coding; to be optimized for reading
            for ci in range(n_vars):
                if col_false_true[ci]:
                    xxi = 'x{}'.format(cj)
                    for iii, xxx in enumerate(X_train_expand_names):
                        if xxi in xxx:
                            if np.abs(U[iii]) > 0.3:
                                new_reg_coeff[ci] = 1.0
                                break
                    cj += 1
            newb.append(new_reg_coeff)

    # Permute variables
    b_ = deepcopy(np.array(newb))
    c_ = deepcopy(c)
    if permutate:
        p = np.random.permutation(n_vars)
        xs[:, :] = xs[:, p]
        b_[:, :] = b_[p, :]
        b_[:, :] = b_[:, p]
        c_[:] = c[p]

    return xs, b_, c_