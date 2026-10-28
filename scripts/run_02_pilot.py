"""Pilot experiment (training gate; configs/pilot.yaml).

Generalization paradigm: the proposed method is trained once for pilot.max_iterations on the training pool with
validation every val_every iterations; the gate compares the best validation epoch with the best deterministic
rule on the same validation instances (data/instances/reference.csv, written by scripts/run_06_run_rules.py):
PASS when the mean fulfilment is at least the rules' mean and the policy reaches the reference on at least
pilot.min_win_share (configs/runs.yaml) of the validation instances. Fails are recorded in docs/pilot-log.md
with the parameter change of the next round (learning rate, entropy coefficient, normalisation, iterations).
Per-instance paradigm (legacy): the listed methods are trained on pilot.instances and the verdict compares the
training incumbents with the rule references. Disabled when configs/pilot.yaml says enabled: false."""
import sys
import time

from _bootstrap import (CONFIGS, ROOT, Job, done, paradigm, pilot_cfg, pilot_enabled, pool_jobs,
                        require_train_confirmed, resolve_workers, run_jobs, runs_cfg, step)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
METHOD_CONFIGS = {"ours": [], "hsddqn": ["configs/baselines/hsddqn.yaml"],
                  "hgnn_ppo": ["configs/baselines/hgnn_ppo.yaml"], "a2c": ["configs/baselines/a2c.yaml"],
                  "dqn": ["configs/baselines/dqn.yaml"], "ddqn": ["configs/baselines/ddqn.yaml"]}
PILOT_DIR = ROOT / "result" / "pilot"
VERDICT = PILOT_DIR / "verdict.txt"

if not pilot_enabled():
    print("[SKIP] configs/pilot.yaml enabled: false (reason in docs/experiment-spec.md)", flush=True)
    sys.exit(0)
if VERDICT.exists() and VERDICT.read_text(encoding="utf-8").startswith("PASS"):
    print(f"[SKIP] pilot already passed: {VERDICT}", flush=True)
    sys.exit(0)

t0 = time.time()
cfg = pilot_cfg()
methods = [str(m) for m in cfg.get("methods", ["ours"])]
max_iterations = int(cfg.get("max_iterations", 200))
require_train_confirmed()
GENERALIZATION = paradigm() == "generalization"
jobs = []
if GENERALIZATION:
    for method in methods:
        jobs += pool_jobs(method, METHOD_CONFIGS.get(method, []), 1, prefix="pilot/",
                          extra_args=["--max-iterations", str(max_iterations)])
    step(f"pilot (generalization): {len(methods)} method(s) x {max_iterations} iterations on the training pool")
else:
    instances = [str(i) for i in cfg.get("instances", [])]
    for method in methods:
        for iid in instances:
            name = f"pilot/{method}/{iid}/run1"
            jobs.append(Job(name, [sys.executable, "train.py", "--config", *CONFIGS, *METHOD_CONFIGS.get(method, []),
                                   "--instances", f"{iid}.txt", "--run-name", name, "--max-iterations",
                                   str(max_iterations), "--seed", "1"],
                            output=ROOT / "result" / name / "train_meta.yaml"))
    step(f"pilot (per_instance): {len(methods)} methods x {len(instances)} instances x {max_iterations} iterations")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))

step("pilot: report and verdict")
import pandas as pd  # noqa: E402

from result.metrics import load_reference  # noqa: E402

reference = load_reference()
PILOT_DIR.mkdir(parents=True, exist_ok=True)
if GENERALIZATION:
    from result.pilot import generalization_verdict  # noqa: E402
    from result.validation import best_epoch_rows, read_val_curve  # noqa: E402

    run_dir = ROOT / "result" / "pilot" / "ours" / "run1"
    rows = read_val_curve(run_dir)
    best_epoch, best_rows = best_epoch_rows(rows)
    if not best_rows:
        sys.exit(f"[FAIL] {run_dir / 'val_curve.csv'} has no rows: set algo.train.val_every > 0 and generate the "
                 "validation pool (scripts/run_01_prepare_data.py)")
    missing = [r["instance_id"] for r in best_rows if r["instance_id"] not in reference]
    if missing:
        print(f"[WARN] {len(missing)} validation instances without a rule reference (run scripts/run_06_run_rules.py "
              f"first): {missing[:5]}", flush=True)
    share = float((runs_cfg().get("pilot") or {}).get("min_win_share", 2 / 3))
    result = generalization_verdict(best_rows, reference, share)
    table = pd.DataFrame([{"instance_id": r["instance_id"], "policy_fulfillment": r["fulfillment_rate"],
                           "reference_fulfillment": (1.0 - reference[r["instance_id"]]) if r["instance_id"] in reference else None}
                          for r in best_rows])
    table.to_csv(PILOT_DIR / "pilot_results.csv", index=False)
    verdict = result["verdict"]
    lines = ["# Pilot report (generalization)", "", f"Best validation epoch: {best_epoch}", "",
             f"Verdict: **{verdict}** ({result['reason']})", "", table.to_markdown(index=False)]
else:
    rows = []
    instances = [str(i) for i in cfg.get("instances", [])]
    for method in methods:
        for iid in instances:
            final = ROOT / "result" / f"pilot/{method}/{iid}/run1" / "fixed_set_final.csv"
            if final.exists():
                record = pd.read_csv(final).iloc[0]
                rows.append({"method": method, "instance_id": iid, "obj_best": float(record["obj_best"]),
                             "obj_greedy_final": float(record["obj_greedy_final"]), "obj_ref": reference.get(iid)})
    table = pd.DataFrame(rows)
    table.to_csv(PILOT_DIR / "pilot_results.csv", index=False)
    ours = table[table["method"] == "ours"] if not table.empty else table
    wins = int(((ours["obj_best"] <= ours["obj_ref"]) & ours["obj_ref"].notna()).sum()) if not ours.empty else 0
    judged = int(ours["obj_ref"].notna().sum()) if not ours.empty else 0
    verdict = "PASS" if judged and wins * 2 >= judged else "FAIL"
    lines = ["# Pilot report", "", f"Verdict: **{verdict}** (ours reaches the best rule on {wins}/{judged} pilot instances)",
             "", table.to_markdown(index=False) if not table.empty else "(no results)"]
(PILOT_DIR / "pilot-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
VERDICT.write_text(f"{verdict}\n", encoding="utf-8")
done(t0, PILOT_DIR / "pilot-report.md", VERDICT)
if verdict != "PASS":
    sys.exit("[FAIL] pilot verdict FAIL: see result/pilot/pilot-report.md and docs/pilot-log.md")
