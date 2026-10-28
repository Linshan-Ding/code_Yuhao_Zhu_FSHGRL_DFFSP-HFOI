"""Instance discovery and selection (moved verbatim from agent/sci_rollout.py)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence


def scan_instance_files(data_dir: str, patterns: Iterable[str]) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        if "isntance" in str(root):
            raise FileNotFoundError(
                f"Training data directory '{root}' does not exist. "
                "Use 'data/instance/competition' instead of 'data/isntance/competition'."
            )
        raise FileNotFoundError(f"Training data directory '{root}' does not exist.")

    files = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    files = sorted({
        path.resolve()
        for path in files
        if not path.name.startswith("._") and path.name != "Introduction.txt"
    })
    if not files:
        raise FileNotFoundError(f"No training instance files matched {list(patterns)} under '{root}'.")
    return files


def resolve_instance_files(data_dir: str, patterns: Iterable[str],
                           instance_names: Sequence[str] | None = None) -> List[Path]:
    all_files = scan_instance_files(data_dir, ["*.txt"] if instance_names else patterns)
    if not instance_names:
        return all_files

    by_name = {path.name: path for path in all_files}
    selected = []
    missing = []
    for name in instance_names:
        key = Path(name).name
        if key in by_name:
            selected.append(by_name[key])
        else:
            missing.append(name)
    if missing:
        available = ", ".join(sorted(by_name.keys()))
        raise FileNotFoundError(
            f"Training instances not found under '{data_dir}': {missing}. "
            f"Available instances: {available}"
        )
    return selected


def select_instance(instance_files: List[Path], strategy: str, index: int) -> tuple[int, Path]:
    if not instance_files:
        raise ValueError("No training instances are available.")
    strategy = (strategy or "fixed").lower()
    if strategy == "random":
        selected_index = random.randrange(len(instance_files))
        return selected_index, instance_files[selected_index]
    if strategy in {"fixed", "round_robin"}:
        selected_index = index % len(instance_files)
        return selected_index, instance_files[selected_index]
    raise ValueError("instance_selection must be one of: fixed, round_robin, random")


# ------------------------------------------------------------- new code: instance folders and index.csv
# One folder per instance: data/instances/<tier>/<instance_id>/<instance_id>.txt + params.yaml. The .txt file
# keeps its original name because train_metrics.csv records it in the instance_name column.
import csv  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from data.instance_io import parse_instance_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INSTANCES_DIR = ROOT / "data" / "instances"
BENCHMARK_DIR = INSTANCES_DIR / "benchmark"
INDEX_PATH = INSTANCES_DIR / "index.csv"
REFERENCE_PATH = INSTANCES_DIR / "reference.csv"
INDEX_COLS = ["instance_id", "tier", "scale", "ddt", "n_orders", "n_products", "n_stages", "n_machines", "source", "path"]
_DDT_PATTERN = re.compile(r"DDT(\d+)")


def instance_id_of(path: str | Path) -> str:
    return Path(path).stem


def parse_ddt(instance_id: str):
    """Due-date tightness encoded in the benchmark file names (DDT500_num50 -> 500); None when absent."""
    match = _DDT_PATTERN.search(str(instance_id))
    return int(match.group(1)) if match else None


def describe_instance(path: str | Path, tier: str = "benchmark", source: str = "external") -> dict:
    """Instance-level parameters for params.yaml (derived from the file; nothing is re-sampled)."""
    path = Path(path)
    spec = parse_instance_file(path)
    arrivals = spec.arrival_times.astype(np.float64)
    due_dates = spec.due_dates.astype(np.float64)
    slack = due_dates - arrivals
    inter_arrival = np.diff(np.sort(arrivals)) if arrivals.size > 1 else np.zeros(0)
    positive_times = spec.process_times[spec.process_times > 0].astype(np.float64)
    orders_per_product = np.bincount(spec.product_types, minlength=spec.product_count)
    gap_mean = float(inter_arrival.mean()) if inter_arrival.size else 0.0
    gap_cv = float(inter_arrival.std() / gap_mean) if inter_arrival.size and gap_mean > 0 else 0.0
    burst, longest = 0, 0
    for gap in inter_arrival:                       # longest run of gaps at most a quarter of the mean gap
        burst = burst + 1 if gap <= 0.25 * gap_mean else 0
        longest = max(longest, burst)
    p_mean = float(positive_times.mean()) if positive_times.size else 0.0
    load_ratio = (p_mean * spec.stage_count) / (gap_mean * spec.machine_count) if gap_mean > 0 else 0.0
    stage_loads = []
    for stage, machines in enumerate(spec.stage_machines):
        stage_times = spec.process_times[:, list(machines)].astype(np.float64)
        stage_positive = stage_times[stage_times > 0]
        if stage_positive.size and gap_mean > 0:
            stage_loads.append(float(stage_positive.mean()) / (gap_mean * len(machines)))
    return {
        "instance_id": path.stem,
        "tier": tier,
        "source": source,
        "file": path.name,
        "n_products": int(spec.product_count),
        "n_stages": int(spec.stage_count),
        "n_machines": int(spec.machine_count),
        "machine_division": [int(v) for v in spec.machine_division],
        "n_orders": int(spec.total_orders),
        "ddt": parse_ddt(path.stem),
        "orders_per_product": [int(v) for v in orders_per_product],
        "arrival_first": float(arrivals.min()) if arrivals.size else 0.0,
        "arrival_last": float(arrivals.max()) if arrivals.size else 0.0,
        "inter_arrival_mean": gap_mean,
        "inter_arrival_cv": gap_cv,
        "inter_arrival_max": float(inter_arrival.max()) if inter_arrival.size else 0.0,
        "max_burst": int(longest + 1) if inter_arrival.size else 0,
        "load_ratio": float(load_ratio),
        "load_ratio_bottleneck": float(max(stage_loads)) if stage_loads else 0.0,
        "due_slack_min": float(slack.min()) if slack.size else 0.0,
        "due_slack_mean": float(slack.mean()) if slack.size else 0.0,
        "due_slack_max": float(slack.max()) if slack.size else 0.0,
        "process_time_min": float(positive_times.min()) if positive_times.size else 0.0,
        "process_time_mean": float(positive_times.mean()) if positive_times.size else 0.0,
        "process_time_max": float(positive_times.max()) if positive_times.size else 0.0,
    }


def write_params_yaml(folder: str | Path, params: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "params.yaml"
    with target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(params, file, sort_keys=False, allow_unicode=True)
    return target


def instance_files_in(folder: str | Path) -> List[Path]:
    """``<folder>/<id>/<id>.txt`` files of one tier directory, sorted."""
    folder = Path(folder)
    return sorted(
        path for path in folder.glob("*/*.txt")
        if path.is_file() and not path.name.startswith("._") and path.stem == path.parent.name
    )


def _existing_params(path: Path) -> dict:
    target = path.parent / "params.yaml"
    if not target.exists():
        return {}
    try:
        return yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def prepare_tier(tier_dir: str | Path, tier: str, source: str) -> List[dict]:
    """params.yaml for every instance folder of one tier and its index rows (nothing is re-sampled).
    Generator-written keys of an existing params.yaml (seed, ddt, slack_rule, ...) are preserved."""
    tier_dir = Path(tier_dir)
    files = instance_files_in(tier_dir)
    if not files:
        raise FileNotFoundError(f"No instance folders found under '{tier_dir}'.")
    rows = []
    for path in files:
        params = describe_instance(path, tier=tier, source=source)
        previous = _existing_params(path)
        for key, value in previous.items():
            if key not in params or (key == "ddt" and params.get("ddt") is None):
                params[key] = value
        write_params_yaml(path.parent, params)
        rows.append({
            "instance_id": params["instance_id"],
            "tier": tier,
            "scale": params["n_orders"],
            "ddt": params["ddt"] if params.get("ddt") is not None else "",
            "n_orders": params["n_orders"],
            "n_products": params["n_products"],
            "n_stages": params["n_stages"],
            "n_machines": params["n_machines"],
            "source": source,
            "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix(),
        })
    return rows


def tier_directories(instances_dir: str | Path = INSTANCES_DIR) -> List[Path]:
    """Tier directories under data/instances/ that hold instance folders (benchmark first, then alphabetical)."""
    instances_dir = Path(instances_dir)
    if not instances_dir.exists():
        return []
    tiers = [p for p in sorted(instances_dir.iterdir()) if p.is_dir() and instance_files_in(p)]
    return sorted(tiers, key=lambda p: (0 if p.name == "benchmark" else 1, p.name))


def prepare_all(instances_dir: str | Path = INSTANCES_DIR, index_path: str | Path = INDEX_PATH) -> List[dict]:
    """Write params.yaml for every tier under ``instances_dir`` and one index (data/instances/index.csv).

    Idempotent: re-running rewrites identical files. ``scale`` is the number of orders (the paper's S) and
    ``ddt`` the due-date tightness (file name or generator record); ``path`` is relative to the repository
    root. Tier order: benchmark, then the generated tiers alphabetically; within a tier by (ddt, scale, id).
    """
    rows = []
    for tier_dir in tier_directories(instances_dir):
        tier = tier_dir.name
        tier_rows = prepare_tier(tier_dir, tier, "external" if tier == "benchmark" else "generated")
        tier_rows.sort(key=lambda row: (row["ddt"] if row["ddt"] != "" else -1, row["scale"], row["instance_id"]))
        rows.extend(tier_rows)
    if not rows:
        raise FileNotFoundError(f"No instance folders found under '{instances_dir}'.")
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_COLS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def prepare_benchmark(benchmark_dir: str | Path = BENCHMARK_DIR, index_path: str | Path = INDEX_PATH,
                      tier: str = "benchmark", source: str = "external") -> List[dict]:
    """Index every tier (see ``prepare_all``) and return the rows of the benchmark tier."""
    rows = prepare_all(Path(benchmark_dir).parent, index_path)
    return [row for row in rows if row["tier"] == tier]


def load_index(index_path: str | Path = INDEX_PATH) -> dict:
    """``{instance_id: row}`` of index.csv (typed scale/ddt); empty when the index has not been written."""
    index_path = Path(index_path)
    if not index_path.exists():
        return {}
    index = {}
    with index_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            row = dict(row)
            for key in ("scale", "n_orders", "n_products", "n_stages", "n_machines"):
                if row.get(key, "") != "":
                    row[key] = int(float(row[key]))
            row["ddt"] = int(float(row["ddt"])) if row.get("ddt", "") != "" else None
            index[row["instance_id"]] = row
    return index


def tier_instance_files(tier: str, instances_dir: str | Path = INSTANCES_DIR) -> List[Path]:
    """Instance files of one tier directory, in index order (then sorted for unindexed files)."""
    tier_dir = Path(instances_dir) / tier
    by_id = {path.stem: path for path in instance_files_in(tier_dir)}
    if not by_id:
        raise FileNotFoundError(f"No instance folders found under '{tier_dir}' (tier '{tier}').")
    index = load_index()
    ordered = [iid for iid, row in index.items() if row.get("tier") == tier and iid in by_id]
    ordered += sorted(iid for iid in by_id if iid not in index)
    return [by_id[iid] for iid in ordered]


def instance_files_by_ids(instance_ids: Sequence[str], instances_dir: str | Path = INSTANCES_DIR) -> List[Path]:
    """Instance files by id or file name, searched across every tier directory."""
    by_id: dict = {}
    for tier_dir in tier_directories(instances_dir):
        for path in instance_files_in(tier_dir):
            by_id.setdefault(path.stem, path)
    missing = [name for name in instance_ids if Path(name).stem not in by_id]
    if missing:
        raise FileNotFoundError(f"Unknown instances {missing}; available tiers: "
                                f"{[p.name for p in tier_directories(instances_dir)]}")
    return [by_id[Path(name).stem] for name in instance_ids]


def benchmark_instance_files(instance_ids: Sequence[str] | None = None,
                             benchmark_dir: str | Path = BENCHMARK_DIR) -> List[Path]:
    """Instance files of the benchmark tier (all, in index order), or the given ids / file names of any tier."""
    if instance_ids:
        return instance_files_by_ids(instance_ids, Path(benchmark_dir).parent)
    return tier_instance_files(Path(benchmark_dir).name, Path(benchmark_dir).parent)


if __name__ == "__main__":   # python -m data.dataset  (scripts/run_01_prepare_data.py)
    _rows = prepare_all()
    _tiers = {}
    for _row in _rows:
        _tiers[_row["tier"]] = _tiers.get(_row["tier"], 0) + 1
    print(f"[DATA] {len(_rows)} instances indexed -> {INDEX_PATH} ({', '.join(f'{k}: {v}' for k, v in _tiers.items())})")
