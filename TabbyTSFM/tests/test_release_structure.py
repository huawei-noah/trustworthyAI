"""Dependency-free checks for the reorganized release contracts."""

from __future__ import annotations

import ast
import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _argparse_options(relative_path: str) -> set[str]:
    tree = ast.parse(_text(relative_path), filename=relative_path)
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    options.add(argument.value)
    return options


def _shell_options_after(relative_path: str, entry_point: str) -> set[str]:
    suffix = _text(relative_path).rsplit(entry_point, 1)[-1]
    return set(re.findall(r"(?<![\w-])--[A-Za-z][A-Za-z0-9_-]*", suffix))


def _literal_assignment(relative_path: str, name: str):
    tree = ast.parse(_text(relative_path), filename=relative_path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {relative_path}")


def _shell_assignment_words(relative_path: str, name: str) -> set[str]:
    match = re.search(rf'(?ms)^{re.escape(name)}="(.*?)"\r?$', _text(relative_path))
    assert match is not None, (relative_path, name)
    value = re.sub(r"\\\r?\n", " ", match.group(1))
    return set(value.split())


def test_all_python_sources_parse() -> None:
    python_files = sorted(
        path
        for base in (ROOT / "src", ROOT / "recipes", ROOT / "benchmarks")
        for path in base.rglob("*.py")
    )
    assert python_files
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_confirmed_pretraining_contract() -> None:
    recipe = _text("recipes/pretrain/train_165k.sh")
    model = _text("src/tabby/models/PatchTSTFM.py")
    hf_config = _text("src/tabby/models/configuration_patchtst_fm.py")

    assert "--total_steps 165000" in recipe
    assert "--d_model 768" in recipe
    assert "--gift_ratio 0" in recipe
    assert "${BLAST_RATIO:?" in recipe
    assert "${CAUKER_V2_RATIO:?" in recipe
    assert "d_model: int = 768" in model
    assert "d_model: int = 768" in hf_config
    assert "n_head: int = 12" in hf_config
    assert "n_layer: int = 20" in hf_config


def test_posttraining_excludes_official_test_windows() -> None:
    data_module = _text("src/tabby/posttraining/data.py")
    env_checker = _text("src/tabby/posttraining/check_env.py")
    train_entry = _text("recipes/posttrain/train.py")

    assert "STRICT_TEST_CUT = True" in data_module
    assert "find_dotenv(usecwd=True)" in data_module
    assert "find_dotenv(usecwd=True)" in env_checker
    assert 'config_dict["strict_test_cut"] = STRICT_TEST_CUT' in train_entry

    test_lengths = _literal_assignment(
        "src/tabby/posttraining/data.py", "GIFTEVAL_TEST_LEN"
    )
    release_tasks = _shell_assignment_words(
        "recipes/posttrain/train_tabby.sh", "long_tasks"
    ) | _shell_assignment_words("recipes/posttrain/train_tabby.sh", "short_tasks")
    assert release_tasks <= set(test_lengths), sorted(release_tasks - set(test_lengths))


def test_time_uses_the_4000_point_visible_context() -> None:
    time_eval = _text("benchmarks/forecasting/time/evaluate.py")

    assert 'p.add_argument("--context_length", type=int, default=4000' in time_eval
    assert "visible_ctx(t[v], args.context_length)" in time_eval


def test_anomaly_release_artifacts_match_the_documented_setup() -> None:
    config = json.loads(_text("benchmarks/anomaly/configs/tabby_tsb_u.json"))
    assert config == {
        "latent_metric": "mahalanobis",
        "latent_stride": None,
        "agg": "mean",
    }

    with (ROOT / "benchmarks/anomaly/results/final_U.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 350
    assert len({row["file_name"] for row in rows}) == 350
    assert all(not row["error"] for row in rows)

    expected = {"VUS-PR": (0.4282, 0.4024), "AUC-PR": (0.3408, 0.2643), "VUS-ROC": (0.8364, 0.9334)}
    for metric, (mean, median) in expected.items():
        values = [float(row[metric]) for row in rows]
        assert round(sum(values) / len(values), 4) == mean
        assert round(statistics.median(values), 4) == median


def test_release_python_has_no_old_package_or_private_mount_paths() -> None:
    old_top_level_packages = {"utils", "models", "data", "tabby_prompt"}
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "recipes").rglob("*.py")) + list((ROOT / "benchmarks").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "tabby_prompt" not in text, path
        assert "/mnt/" not in text, path
        assert "C:\\Users\\" not in text, path
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                assert node.module.split(".", 1)[0] not in old_top_level_packages, path
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] not in old_top_level_packages, path


def test_shell_recipe_options_exist_in_their_entry_points() -> None:
    cases = (
        (
            "recipes/pretrain/train_165k.sh",
            "recipes/pretrain/train.py",
            "recipes/pretrain/train.py",
        ),
        (
            "recipes/posttrain/train_tabby.sh",
            "recipes/posttrain/train.py",
            "recipes/posttrain/train.py",
        ),
        (
            "benchmarks/forecasting/gift_eval/evaluate.sh",
            "benchmarks/forecasting/gift_eval/evaluate.py",
            "benchmarks/forecasting/gift_eval/evaluate.py",
        ),
    )
    for shell_path, python_path, entry_point in cases:
        missing = _shell_options_after(shell_path, entry_point) - _argparse_options(
            python_path
        )
        assert not missing, (shell_path, sorted(missing))


if __name__ == "__main__":
    checks = [
        test_all_python_sources_parse,
        test_confirmed_pretraining_contract,
        test_posttraining_excludes_official_test_windows,
        test_time_uses_the_4000_point_visible_context,
        test_anomaly_release_artifacts_match_the_documented_setup,
        test_release_python_has_no_old_package_or_private_mount_paths,
        test_shell_recipe_options_exist_in_their_entry_points,
    ]
    for check in checks:
        check()
        print(f"[OK] {check.__name__}")
