"""Run every stage in dependency order: data -> pilot (gate; skipped when disabled) -> training (ours,
ablations, learned baselines) in parallel with the CPU rule baselines and the exact MILP check -> evaluation
and the mask-threshold sensitivity -> aggregation.
An optional integer argument caps the concurrency of every stage (1 = fully serial)."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARG = [a for a in sys.argv[1:] if a.isdigit()][:1]
SERIAL = ARG == ["1"]

STAGES = [
    ("data", ["run_01_prepare_data.py"], []),
    ("pilot", ["run_02_pilot.py"], ["data"]),
    ("train", ["run_03_train_ours.py", "run_04_train_ablations.py", "run_05_train_learned_baselines.py"], ["pilot"]),
    ("rules", ["run_06_run_rules.py"], ["data"]),
    ("milp", ["run_08_milp_check.py"], ["data"]),
    ("eval", ["run_07_eval_main.py"], ["train", "rules"]),
    ("sensitivity", ["run_10_sensitivity.py"], ["train", "milp"]),
    ("aggregate", ["run_09_aggregate_stats.py"], ["eval", "milp", "sensitivity"]),
]

done_stages, running, t0, failed = set(), [], time.time(), []
todo = {name: (scripts, set(deps)) for name, scripts, deps in STAGES}
while todo or running:
    for name in [n for n, (_, deps) in todo.items() if deps <= done_stages]:
        if failed or (SERIAL and running):
            break
        scripts, _ = todo.pop(name)
        print("\n" + "#" * 72 + f"\n# stage {name}: {', '.join(scripts)}\n" + "#" * 72, flush=True)
        if SERIAL:
            procs = []
            for script in scripts:
                proc = subprocess.Popen([sys.executable, str(HERE / script)] + ARG)
                proc.wait()
                procs.append((script, proc))
        else:
            procs = [(script, subprocess.Popen([sys.executable, str(HERE / script)] + ARG)) for script in scripts]
        running.append((name, procs, time.time()))
    time.sleep(1)
    for entry in list(running):
        name, procs, started = entry
        if any(p.poll() is None for _, p in procs):
            continue
        running.remove(entry)
        bad = [s for s, p in procs if p.returncode != 0]
        if bad:
            failed += bad
            print(f"[FAIL] stage {name}: {bad} failed", flush=True)
        else:
            done_stages.add(name)
            print(f"[STAGE OK] {name} in {(time.time() - started) / 60:.1f} min", flush=True)
    if failed and not running:
        sys.exit(f"[FAIL] {failed}; later stages cancelled")
print(f"\n[ALL OK] all experiment data produced in {(time.time() - t0) / 60:.1f} min, see result/", flush=True)
