"""Exact MILP check (baselines/milp_exact.py): model, verifier, simulator replay, brute force, output contract."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.milp_exact import (RESULT_COLS, Schedule, brute_force, build_model, build_problem,  # noqa: E402
                                  extract_schedule, left_shift, replay, resolve_backend, solve, solve_instance,
                                  stage_schedule_count, verify, write_results)
from data.instance_io import parse_instance_file  # noqa: E402

TINY = ROOT / "data" / "instances" / "tiny" / "TNY5000_DDT500_num4" / "TNY5000_DDT500_num4.txt"
TINY5 = ROOT / "data" / "instances" / "tiny5" / "TNF5500_DDT500_num6" / "TNF5500_DDT500_num6.txt"


def _write(tmp_path: Path, name: str, lines) -> Path:
    path = tmp_path / f"{name}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def micro(tmp_path):
    """1 product, 2 stages, 1 machine each (p = 10 and 5); three orders arriving at 0, 0, 2 with due dates
    15, 25, 40: at most two are on time (the single machine of stage 1 serialises them: completions 10, 20, 30;
    stage 2 adds 5 -> 15, 25, 35) and the third order is on time only if processed first... every order of
    the first two is on time in the order (1, 2, 3): 15 <= 15, 25 <= 25, 35 <= 40 -> all three on time."""
    return _write(tmp_path, "MICRO_num3", ["1 2 2", "1 1", "10 5", "1 2 3", "0 0 0", "0 0 2", "15 25 40"])


@pytest.fixture
def hopeless(tmp_path):
    """As micro, but the second order is hopeless (due 12 < earliest completion 25 when processed second,
    and processing it first delays the others): the optimum drops one order, MILP-A still processes it."""
    return _write(tmp_path, "HOPELESS_num3", ["1 2 2", "1 1", "10 5", "1 2 3", "0 0 0", "0 0 2", "15 12 40"])


def test_problem_and_model_sizes(micro):
    problem = build_problem(parse_instance_file(micro))
    assert problem.n_orders == 3 and len(problem.ops) == 6 and problem.stages == 2
    assert problem.c_ub == pytest.approx(2 + 3 * 15) and problem.big_l == pytest.approx(problem.c_ub + 10)
    model = build_model(problem, "A")
    assert len(model.x_index) == 6 and len(model.z_index) == 3 and len(model.c_index) == 6
    assert model.n_pairs == 2 * 3 and model.n_binary == 6 + 6 + 3        # 3 ops per machine -> 3 pairs each
    assert len(model.rows) == 6 + 6 + 2 * 6 + 3                          # (2), (3)+(4), (6)+(7), (8)
    with pytest.raises(ValueError):
        build_model(problem, "C")


def test_micro_optimum_both_variants_verify_replay_and_brute_force(micro):
    problem = build_problem(parse_instance_file(micro))
    for variant in ("A", "B"):
        model = build_model(problem, variant)
        result = solve(model, resolve_backend("highs"), 30.0)
        assert result["status"] == "optimal" and result["objective"] == pytest.approx(3.0)
        schedule = extract_schedule(problem, model, result["values"])
        assert schedule.sequences[0] == [(0, 0), (1, 0), (2, 0)] and schedule.sequences[1] == [(0, 1), (1, 1), (2, 1)]
        verified = verify(problem, schedule, result["objective"])
        assert verified["on_time"] == 3 and verified["times"][(2, 1)] == (30.0, 35.0)
        replayed = replay(problem, schedule)
        assert replayed["eta"] == pytest.approx(1.0) and replayed["on_time"] == [0, 1, 2]
        assert replayed["starts"][(2, 1)] == 30.0                         # the simulator reproduces the timing
        exhaustive = brute_force(problem, variant)
        assert exhaustive["on_time"] == 3 and exhaustive["nodes"] > 0 and exhaustive["status"] == "ok"


def test_hopeless_order_is_abandoned_by_b_and_processed_late_by_a(hopeless):
    problem = build_problem(parse_instance_file(hopeless))
    values = {}
    for variant in ("A", "B"):
        model = build_model(problem, variant)
        result = solve(model, "highs", 30.0)
        assert result["status"] == "optimal" and result["objective"] == pytest.approx(2.0)
        schedule = extract_schedule(problem, model, result["values"])
        assert sorted(schedule.on_time) == [0, 2]
        if variant == "A":
            assert len(schedule.assignment) == 6                          # every operation processed
        else:
            assert len(schedule.assignment) == 4 and 1 not in schedule.processed_orders
        verify(problem, schedule, result["objective"])
        replayed = replay(problem, schedule)
        assert replayed["on_time"] == [0, 2] and replayed["discarded"] == 1  # the simulator discards order 1
        values[variant] = brute_force(problem, variant)["on_time"]
    assert values == {"A": 2, "B": 2}


def test_verifier_rejects_cyclic_sequences_and_false_claims(micro):
    problem = build_problem(parse_instance_file(micro))
    assignment = {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 1}
    cyclic = Schedule(assignment, {0: [(1, 0), (0, 0)], 1: [(0, 1), (1, 1)]},
                      {op: 0.0 for op in assignment}, [0, 1])
    times = left_shift(problem, cyclic)                                   # order 1 first on machine 0 is fine
    assert times[(1, 0)] == (0.0, 10.0) and times[(0, 1)] == (20.0, 25.0)
    with pytest.raises(AssertionError):                                   # claimed completions earlier than feasible
        verify(problem, cyclic, 2.0)
    broken = Schedule({(0, 1): 1}, {1: [(0, 1)]}, {(0, 1): 100.0}, [])
    with pytest.raises(ValueError):                                       # predecessor not processed
        left_shift(problem, broken)


def test_tiny_instance_end_to_end_and_output_contract(tmp_path):
    rows = solve_instance(TINY, ("A", "B"), "highs", 60.0, brute_force_max_orders=4, tier="tiny")
    assert [row["variant"] for row in rows] == ["A", "B"]
    for row in rows:
        assert row["status"] == "optimal" and row["replay_ok"] is True and row["brute_force_ok"] is True
        assert row["brute_force_status"] == "ok" and row["brute_force_nodes"] > 0
        assert row["completed_opt"] == 3 and row["eta_opt"] == pytest.approx(0.75)
        assert row["verifier_eta"] == row["replay_eta"] == row["brute_force_eta"] == pytest.approx(0.75)
        assert row["S"] == 4 and row["n_ops"] == 12 and row["n_binary"] == 64 and row["backend"] == "highs"
        assert row["big_L"] > row["c_ub"] > 0 and row["gap"] == 0.0
    assert rows[0]["completed_opt"] == rows[1]["completed_opt"]           # MILP-A and MILP-B agree
    out = write_results(rows, tmp_path / "milp" / "milp_results.csv")
    import pandas as pd

    table = pd.read_csv(out)
    assert list(table.columns) == RESULT_COLS and len(table) == 2
    assert np.allclose(table["eta_opt"] * table["S"], table["completed_opt"])  # S * eta is an integer


def test_brute_force_gates(micro):
    problem = build_problem(parse_instance_file(micro))
    assert stage_schedule_count(problem, [0, 1, 2]) == 6                   # 3 orders, one machine: 3! sequences
    tiny5 = build_problem(parse_instance_file(TINY5))
    assert stage_schedule_count(tiny5, list(range(6))) == 720 * 210        # 6! x C(10, 4): far beyond the gate
    assert brute_force(tiny5, "A")["status"] == "skipped_size"
    budget = brute_force(problem, "B", max_nodes=5)
    assert budget["status"] == "budget" and budget["on_time"] is None and budget["nodes"] == 6
    rows = solve_instance(TINY5, ("B",), "highs", 60.0, brute_force_max_orders=6, tier="tiny5")
    assert rows[0]["status"] == "optimal" and rows[0]["brute_force_status"] == "skipped_size"
    assert rows[0]["brute_force_ok"] == "" and rows[0]["brute_force_eta"] == ""


def test_backend_resolution():
    assert resolve_backend("highs") == "highs" and resolve_backend("scipy") == "highs"
    assert resolve_backend("auto") in ("highs", "gurobi")
    with pytest.raises(ValueError):
        resolve_backend("cplex")
