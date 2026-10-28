"""Common prelude of the zero-argument entry points ``scripts/run_XX_*.py`` (new code, after the
opt-paper-codegen runner template): anchors the project root, timing / skipping helpers, single and
concurrent job execution with resource clamping, and readers of configs/runs.yaml, parallel.yaml, pilot.yaml.

Each job runs in its own subprocess with the BLAS / torch thread count pinned (``threads_per_job`` of
configs/parallel.yaml, default: cores // workers), writes its own ``result/<name>/stdout.log`` and is
skipped when its product already exists, so every script can be re-run after an interruption.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAFETY = 0.9              # share of the free GPU memory counted as budget
POLL = 0.5                # seconds between polls of running jobs
PROGRESS_EVERY = 300      # seconds between progress tables
CONFIGS = ["configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml"]   # the default trio
if (ROOT / "configs" / "paper.yaml").exists():
    CONFIGS.append("configs/paper.yaml")                                          # the paper layer (v2)


# ------------------------------------------------------------------ console helpers

def step(title: str) -> None:
    print("\n" + "=" * 72 + f"\n[RUN] {title}\n" + "=" * 72, flush=True)


def done(t0: float, *outputs) -> None:
    print(f"[OK] {time.time() - t0:.1f} s", flush=True)
    for item in outputs:
        path = Path(item)
        print(f"     product {path.resolve()} {'[written]' if path.exists() else '[MISSING]'}", flush=True)


def skip_if_exists(path, what: str = "") -> bool:
    path = Path(path)
    if path.exists():
        print(f"[SKIP] {what or path.name} already done: {path}", flush=True)
        return True
    return False


def run_py(entry: str, *args, env: dict | None = None) -> None:
    """Run a script (``x.py``) or a module (``pkg.mod``) with the project interpreter; exit on failure."""
    head = [entry] if entry.endswith(".py") else ["-m", entry]
    cmd = [sys.executable] + head + [str(a) for a in args]
    print("[CMD] " + " ".join(cmd), flush=True)
    if subprocess.call(cmd, cwd=str(ROOT), env=env or job_env(None)) != 0:
        sys.exit(f"[FAIL] {entry} exited with a non-zero code; later steps cancelled")


def merge_csv(parts, out) -> int:
    """Concatenate CSV files with a common column set (missing files are skipped)."""
    import pandas as pd

    frames = [pd.read_csv(p) for p in parts if Path(p).exists() and Path(p).stat().st_size > 0]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        print(f"[WARN] nothing to merge into {out}", flush=True)
        return 0
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(out, index=False)
    print(f"[MERGE] {len(frames)} files, {len(merged)} rows -> {out}", flush=True)
    return len(merged)


# ------------------------------------------------------------------- configuration

def read_yaml(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def runs_cfg() -> dict:
    return read_yaml(ROOT / "configs" / "runs.yaml")


def n_runs(key: str, default: int = 3) -> int:
    """Independent runs per method class (configs/runs.yaml: ours / ablations / learned_baselines / rules_random)."""
    value = runs_cfg().get(key, default)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def seed_args(n: int) -> list:
    """A single run is the one exception to the no-seed policy: its seed is fixed and recorded."""
    return ["--seed", "1"] if n == 1 else []


def parallel_cfg() -> dict:
    return read_yaml(ROOT / "configs" / "parallel.yaml")


def requested_workers(default: int) -> int:
    """Concurrency: optional integer argument > configs/parallel.yaml max_concurrent > script constant."""
    for arg in sys.argv[1:]:
        if arg.isdigit():
            return max(1, int(arg))
    value = parallel_cfg().get("max_concurrent")
    if value is not None:
        return max(1, int(value))
    return max(1, int(default))


def threads_per_job(workers: int) -> int:
    value = parallel_cfg().get("threads_per_job")
    if value:
        return max(1, int(value))
    return max(1, (os.cpu_count() or 2) // max(workers, 1))


def job_env(threads: int | None) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if threads:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(threads)
    return env


def instance_ids(*tiers) -> list:
    """Instance ids of the given tiers from data/instances/index.csv (run_01 writes it)."""
    index = ROOT / "data" / "instances" / "index.csv"
    if not index.exists():
        sys.exit("[FAIL] data/instances/index.csv missing: run python scripts/run_01_prepare_data.py first")
    with index.open(encoding="utf-8") as file:
        return [row["instance_id"] for row in csv.DictReader(file) if not tiers or row["tier"] in tiers]


def paradigm() -> str:
    """``instance.paradigm`` of the merged configuration the scripts train with (CONFIGS)."""
    from configs.config import load_config

    return str(load_config([ROOT / c for c in CONFIGS]).instance.paradigm).lower()


def eval_tiers_cfg() -> list:
    from configs.config import load_config

    return [str(t) for t in load_config([ROOT / c for c in CONFIGS]).instance.eval_tiers]


def collector_workers_cfg() -> int:
    """env.collector_workers of configs/env.yaml: environment worker processes each training job starts."""
    try:
        return max(0, int((read_yaml(ROOT / "configs" / "env.yaml").get("env") or {}).get("collector_workers", 0) or 0))
    except (TypeError, ValueError):
        return 0


def pilot_cfg() -> dict:
    return read_yaml(ROOT / "configs" / "pilot.yaml")


def pilot_enabled() -> bool:
    return bool(pilot_cfg().get("enabled", False))


def require_train_confirmed() -> None:
    """No training job starts before the training-parameter table is confirmed (configs/algo.yaml)."""
    cfg = ROOT / "configs" / "algo.yaml"
    match = re.search(r"^\s*train_confirmed\s*:\s*(\w+)", cfg.read_text(encoding="utf-8"), re.M) if cfg.exists() else None
    if not (match and match.group(1).lower() == "true"):
        sys.exit("[FAIL] training-parameter table not confirmed: review configs/algo.yaml and set "
                 "algo.train_confirmed: true (record train_version)")


# ---------------------------------------------------------------- resources

def free_vram_gb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / 1024 ** 3
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return max(float(x) for x in out.stdout.split()) / 1024
    except Exception:
        pass
    return None


def free_ram_gb():
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        return int(info["MemAvailable"].strip().split()[0]) / 1024 ** 2
    except Exception:
        return None


def resolve_workers(default: int, resource: str = "cpu", gpu_mem_per_job_gb=None, env_workers: int | None = None,
                    parallel: bool = True) -> int:
    """Clamp the requested concurrency to what the machine supports and say why (never silently).
    ``env_workers`` (environment worker processes per job) defaults to configs/env.yaml collector_workers."""
    if env_workers is None:
        env_workers = collector_workers_cfg()
    want = requested_workers(default)
    if not parallel:
        print(f"[PARALLEL] this script runs serially (PARALLEL=False; request {want} ignored)", flush=True)
        return 1
    cap, why = want, "below the device limits"
    if resource == "gpu":
        free = free_vram_gb()
        if free is None:
            cap, why = 1, "no GPU memory information (no GPU or nvidia-smi unavailable): serial"
        elif gpu_mem_per_job_gb:
            fit = max(1, int(free * SAFETY // gpu_mem_per_job_gb))
            if fit < want:
                cap, why = fit, f"free GPU memory {free:.1f} GB x {SAFETY:.0%} / {gpu_mem_per_job_gb} GB per job"
    cores = os.cpu_count() or 2
    fit = max(1, cores - env_workers - 1) if resource == "cpu" else max(1, cores - env_workers)
    if fit < cap:
        cap, why = fit, f"{cores} cores - {env_workers} environment workers - 1"
    ram = free_ram_gb()
    if ram is not None and ram < 4:
        cap, why = 1, f"only {ram:.1f} GB RAM available: serial"
    if cap < want:
        print(f"[PARALLEL] requested {want}, clamped to {cap}: {why}", flush=True)
    else:
        print(f"[PARALLEL] {cap} concurrent job(s) ({why})", flush=True)
    return cap


# ---------------------------------------------------------------- concurrent jobs

class Job:
    """A concurrent job: name, command, product (skip when present), log path."""

    def __init__(self, name: str, cmd, output=None):
        self.name, self.cmd, self.output = name, [str(c) for c in cmd], output
        self.log = ROOT / "result" / name / "stdout.log"
        self.rc, self.t0, self.wall, self.oom_retries = None, None, 0.0, 0


OOM_PAT = re.compile(r"out of memory|outofmemoryerror|\boom\b|killed process|cuda error: out of memory", re.I)


def _is_oom(job: Job) -> bool:
    if not job.log.exists():
        return False
    return bool(OOM_PAT.search(job.log.read_text(errors="replace")[-4000:]))


def run_jobs(jobs, workers: int, gpu_mem_per_job_gb=None, fail_fast: str = "drain", threads: int | None = None) -> list:
    """Run independent jobs ``workers`` at a time; each writes result/<name>/stdout.log. A failure stops the
    launch of new jobs (running ones drain). Out-of-memory failures lower the concurrency by one and re-queue
    the job (twice at most); the floor is one job, so the worst case is a serial run."""
    pending = [j for j in jobs if not (j.output and Path(j.output).exists())]
    for job in jobs:
        if job not in pending:
            print(f"[SKIP] {job.name} already done", flush=True)
    if not pending:
        return []
    threads = threads or threads_per_job(workers)
    plog, running, finished, t_start = ROOT / "result" / "parallel-log.md", [], [], time.time()
    plog.parent.mkdir(parents=True, exist_ok=True)
    stop_new, last_report = False, 0.0

    def launch(job: Job) -> None:
        job.log.parent.mkdir(parents=True, exist_ok=True)
        job.t0 = time.time()
        handle = job.log.open("w", encoding="utf-8")
        proc = subprocess.Popen(job.cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, env=job_env(threads))
        running.append((job, proc, handle))
        print(f"[START] {job.name} -> {job.log.relative_to(ROOT)}", flush=True)

    while pending or running:
        while pending and not stop_new and len(running) < workers:
            if gpu_mem_per_job_gb:
                free = free_vram_gb()
                if free is not None and free * SAFETY < gpu_mem_per_job_gb and running:
                    break
            launch(pending.pop(0))
        time.sleep(POLL)
        for entry in list(running):
            job, proc, handle = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            handle.close()
            job.rc, job.wall = proc.returncode, time.time() - job.t0
            if job.rc != 0 and _is_oom(job) and job.oom_retries < 2:
                job.oom_retries += 1
                workers = max(1, workers - 1)
                pending.insert(0, job)
                message = f"- {job.name}: out of memory, concurrency lowered to {workers}, re-queued (retry {job.oom_retries})"
                print(f"[OOM] {message[2:]}", flush=True)
                with plog.open("a", encoding="utf-8") as file:
                    file.write(message + "\n")
                continue
            finished.append(job)
            tag = "OK" if job.rc == 0 else "FAIL"
            print(f"[{tag}] {job.name} {job.wall / 60:.1f} min (exit code {job.rc})", flush=True)
            if job.rc != 0 and fail_fast:
                stop_new = True
                if pending:
                    print(f"[CANCEL] {len(pending)} job(s) not started: {[j.name for j in pending]}", flush=True)
                    pending.clear()
        if running and time.time() - last_report >= PROGRESS_EVERY:
            last_report = time.time()
            bar = "  ".join(f"{j.name}:{(time.time() - j.t0) / 60:.0f}min" for j, _, _ in running)
            print(f"[RUNNING {len(running)}/{workers}] {bar}   pending {len(pending)}", flush=True)

    serial = sum(j.wall for j in finished)
    total = time.time() - t_start
    print(f"\n[PARALLEL] {len(finished)} job(s): serial sum {serial / 60:.1f} min, wall {total / 60:.1f} min, "
          f"speed-up {serial / max(total, 1e-9):.2f}x", flush=True)
    bad = [j for j in finished if j.rc != 0]
    if bad:
        sys.exit(f"[FAIL] {len(bad)} job(s) failed: {[j.name for j in bad]}; see result/<name>/stdout.log")
    return finished


def pool_jobs(method: str, extra_configs, runs: int, prefix: str = "", extra_args=()) -> list:
    """Generalisation paradigm: one job per run -> result/<prefix><method>/run<k>/ (the training pool, the
    instance selection and the validation set come from the configuration)."""
    jobs = []
    for k in range(1, runs + 1):
        name = f"{prefix}{method}/run{k}"
        jobs.append(Job(name, [sys.executable, "train.py", "--config", *CONFIGS, *extra_configs,
                               "--run-name", name, *seed_args(runs), *extra_args],
                        output=ROOT / "result" / name / "train_meta.yaml"))
    return jobs


def method_jobs(method: str, extra_configs, instance_list, runs: int, prefix: str = "") -> list:
    """Jobs of a method under the configured paradigm (per-instance matrix or one job per run)."""
    if paradigm() == "generalization":
        return pool_jobs(method, extra_configs, runs, prefix)
    return training_jobs(method, extra_configs, instance_list, runs, prefix)


def training_jobs(method: str, extra_configs, instance_list, runs: int, prefix: str = "") -> list:
    """Per-instance training matrix: one job per (instance, run) -> result/<prefix><method>/<id>/run<k>/."""
    jobs = []
    for iid in instance_list:
        for k in range(1, runs + 1):
            name = f"{prefix}{method}/{iid}/run{k}"
            jobs.append(Job(name, [sys.executable, "train.py", "--config", *CONFIGS, *extra_configs,
                                   "--instances", f"{iid}.txt", "--run-name", name, *seed_args(runs)],
                            output=ROOT / "result" / name / "train_meta.yaml"))
    return jobs
