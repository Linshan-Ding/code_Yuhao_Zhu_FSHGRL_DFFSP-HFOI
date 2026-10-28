"""The zero-argument entry points compile, declare their concurrency constants and read the config files."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_scripts_compile_and_declare_concurrency():
    names = sorted(p.name for p in SCRIPTS.glob("run_*.py"))
    assert names == ["run_00_smoke.py", "run_01_prepare_data.py", "run_02_pilot.py", "run_03_train_ours.py",
                     "run_04_train_ablations.py", "run_05_train_learned_baselines.py", "run_06_run_rules.py",
                     "run_07_eval_main.py", "run_09_aggregate_stats.py", "run_all.py"]
    for name in names:
        tree = ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))
        assigned = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names = target.elts if isinstance(target, ast.Tuple) else [target]
                    assigned |= {n.id for n in names if isinstance(n, ast.Name)}
        if name in ("run_02_pilot.py", "run_03_train_ours.py", "run_04_train_ablations.py",
                    "run_05_train_learned_baselines.py", "run_07_eval_main.py"):
            assert {"PARALLEL", "RESOURCE", "MAX_CONCURRENT"} <= assigned, name


def test_bootstrap_reads_configs(monkeypatch):
    import _bootstrap

    monkeypatch.setattr(sys, "argv", ["x"])
    assert _bootstrap.n_runs("ours", 3) == 3 and _bootstrap.n_runs("learned_baselines", 1) == 1
    assert _bootstrap.n_runs("missing_key", 7) == 7
    assert _bootstrap.seed_args(1) == ["--seed", "1"] and _bootstrap.seed_args(3) == []
    assert _bootstrap.requested_workers(5) == 2          # configs/parallel.yaml max_concurrent
    monkeypatch.setattr(sys, "argv", ["x", "4"])
    assert _bootstrap.requested_workers(5) == 4
    assert _bootstrap.pilot_enabled() is True
    assert len(_bootstrap.instance_ids("benchmark")) == 15
    jobs = _bootstrap.training_jobs("ours", [], ["DDT500_num50"], 2)
    assert [j.name for j in jobs] == ["ours/DDT500_num50/run1", "ours/DDT500_num50/run2"]
    assert "--seed" not in jobs[0].cmd and "--instances" in jobs[0].cmd


def test_readme_commands_are_copy_safe():
    import re

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    pattern = re.compile(r"(\bdo\b|\bdone\b|\$[A-Za-z_{]|&&|[^&]& *$|\.\.\.|\\$|export |python3)", re.M)
    assert not pattern.findall(text)
    blocks = re.findall(r"```\n(.*?)```", text, re.S)
    commands = [line.strip() for block in blocks for line in block.splitlines() if line.strip()]
    for command in commands:
        assert command.startswith(("python ", "pip ")), command
        if command.startswith("python scripts/"):
            assert (ROOT / command.split()[1]).exists(), command
