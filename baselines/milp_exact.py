"""Exact MILP solutions on small instances (v2, new code): verification of the paper's model.

    python baselines/milp_exact.py                                  # tiers tiny and tiny5, both variants
    python baselines/milp_exact.py --instances TNY5000_DDT500_num4 --variants B --time-limit 60

The model is the MILP of Section 3.2 (objective (1), assignment (2), arrival (3), precedence (4), the
disjunctive machine constraints (6)-(7) on the pair sets (5), the due-date indicator (8), domains (9)) in two
variants:

* MILP-A, the formulation of the paper: every operation of every order is processed (constraint (2));
* MILP-B, the abandonment variant: sum_m x_sjm = z_s, so an order that is not completed on time occupies no
  capacity. This is the convention of the online simulator, which discards an order that cannot meet its
  due date. The two optima coincide (late operations can always be appended after the on-time work, so an
  optimal MILP-B schedule extends to a MILP-A schedule with the same objective, and an optimal MILP-A schedule
  restricted to its on-time orders is MILP-B feasible); the solver confirms this on every instance.

Big-L: C_UB = max_s a_s + sum_(s,j) max_m p_sjm bounds every left-shifted completion time, every C_sj gets
the explicit upper bound C_UB and L = C_UB + max p; both are recorded. The solver is HiGHS through
scipy.optimize.milp (exact, no licence); the original Gurobi model stays as an explicit, unsupported option
(--backend gurobi).

Checks on every solution:
* verifier: the machine assignment and machine sequences of the solution are re-timed from scratch by the
  left-shift rule (start = max(arrival or previous completion, machine free)); the on-time count under the
  simulator's convention (C <= d) must be >= the MILP objective (equal for MILP-B);
* simulator replay: ``environment/simulator.py`` runs the same assignment and sequences through its event
  loop (arrivals, discard at the due date, discard at a start that would finish late, machine release); the
  replayed fulfilment must equal the optimum for MILP-B and be >= it for MILP-A (the simulator discards
  MILP-A's late operations, which frees capacity); the replay keeps every candidate order (order_top_k 0);
* brute force (S <= --brute-force-max-orders): exhaustive enumeration of machine assignments and machine
  sequences, stage by stage with left-shift timing and a bound, over every subset of processed orders; its
  optimum must equal the MILP optimum of both variants. The enumeration is attempted only when every stage
  has at most --brute-force-max-stage-schedules sequenced assignments (n! x C(n + m - 1, m - 1) for n orders
  on m machines: 5040 for the tiny structure at S = 6, 151 200 for the benchmark structure) and is abandoned
  after --brute-force-max-nodes search nodes (brute_force_status: ok | mismatch | skipped_size | budget).

Output ``result/milp/milp_results.csv``, one row per instance x variant: instance, tier, S, n_ops (|K|),
n_binary, n_pairs (sequencing pairs), variant, completed_opt, eta_opt, status, gap, solve_s, backend,
big_L, c_ub, time_limit_s, verifier_eta, replay_eta, replay_ok, brute_force_eta, brute_force_ok,
brute_force_nodes, brute_force_s.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import instance_files_by_ids, load_index, tier_instance_files  # noqa: E402
from data.instance_io import InstanceSpec, parse_instance_file  # noqa: E402
from environment.simulator import TrainingSimulator  # noqa: E402

VARIANTS = ("A", "B")
DEFAULT_TIERS = ("tiny", "tiny5")
DEFAULT_OUT = ROOT / "result" / "milp" / "milp_results.csv"
RESULT_COLS = ["instance", "tier", "S", "n_ops", "n_binary", "n_pairs", "variant", "completed_opt", "eta_opt",
               "status", "gap", "solve_s", "backend", "big_L", "c_ub", "time_limit_s", "verifier_eta", "replay_eta",
               "replay_ok", "brute_force_eta", "brute_force_ok", "brute_force_status", "brute_force_nodes",
               "brute_force_s"]
BRUTE_FORCE_MAX_STAGE_SCHEDULES = 10_000
BRUTE_FORCE_MAX_NODES = 50_000_000
REPLAY_PARAS = {"order_top_k": 0, "fluid_mode": "off", "discard_penalty": 1.0, "progress_reward_coeff": 0.0}
EPS = 1e-6


# -- problem data ---------------------------------------------------------------------------------------------
@dataclass
class Problem:
    """The instance as the MILP sees it: operations (s, j), eligible machines and processing times."""
    spec: InstanceSpec
    ops: List[Tuple[int, int]]                       # (order, stage) in route order
    eligible: Dict[Tuple[int, int], List[int]]       # op -> eligible machines
    ptime: Dict[Tuple[int, int, int], float]         # (order, stage, machine) -> processing time
    c_ub: float = 0.0                                 # upper bound of every completion time
    big_l: float = 0.0

    @property
    def n_orders(self) -> int:
        return self.spec.total_orders

    @property
    def stages(self) -> int:
        return self.spec.stage_count

    def arrival(self, order: int) -> float:
        return float(self.spec.arrival_times[order])

    def due(self, order: int) -> float:
        return float(self.spec.due_dates[order])

    def task(self, order: int, stage: int) -> int:
        return int(self.spec.product_types[order]) * self.spec.stage_count + stage


def build_problem(spec: InstanceSpec) -> Problem:
    ops, eligible, ptime = [], {}, {}
    for order in range(spec.total_orders):
        product = int(spec.product_types[order])
        for stage in range(spec.stage_count):
            machines = []
            for machine in spec.stage_machines[stage]:
                p = float(spec.process_times[product, machine])
                if p > 0:
                    machines.append(int(machine))
                    ptime[(order, stage, int(machine))] = p
            if not machines:
                raise ValueError(f"{spec.path.name}: order {order} stage {stage} has no eligible machine")
            ops.append((order, stage))
            eligible[(order, stage)] = machines
    p_max = {op: max(ptime[(op[0], op[1], m)] for m in eligible[op]) for op in ops}
    c_ub = float(spec.arrival_times.max()) + sum(p_max.values()) if ops else 0.0
    big_l = c_ub + (max(p_max.values()) if p_max else 0.0)
    return Problem(spec, ops, eligible, ptime, c_ub, big_l)


# -- the MILP --------------------------------------------------------------------------------------------------
@dataclass
class Model:
    """Sparse matrix form max c'v s.t. lb <= A v <= ub, bounds, integrality (one layout for both solvers)."""
    n_vars: int
    objective: np.ndarray
    rows: List[Tuple[Dict[int, float], float, float]] = field(default_factory=list)   # (coefficients, lb, ub)
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    integrality: Optional[np.ndarray] = None
    x_index: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    y_index: Dict[Tuple[Tuple[int, int], Tuple[int, int], int], int] = field(default_factory=dict)
    z_index: Dict[int, int] = field(default_factory=dict)
    c_index: Dict[Tuple[int, int], int] = field(default_factory=dict)

    @property
    def n_binary(self) -> int:
        return len(self.x_index) + len(self.y_index) + len(self.z_index)

    @property
    def n_pairs(self) -> int:
        return len(self.y_index)

    def add_row(self, coefficients: Dict[int, float], lb: float, ub: float) -> None:
        self.rows.append((coefficients, lb, ub))


def build_model(problem: Problem, variant: str) -> Model:
    """Constraints (2)-(9) of the paper; variant B replaces the right-hand side 1 of (2) by z_s."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    x_index, y_index, z_index, c_index = {}, {}, {}, {}
    n = 0
    for op in problem.ops:
        for machine in problem.eligible[op]:
            x_index[(op[0], op[1], machine)] = n
            n += 1
    machine_ops: Dict[int, List[Tuple[int, int]]] = {}
    for op in problem.ops:
        for machine in problem.eligible[op]:
            machine_ops.setdefault(machine, []).append(op)
    for machine in sorted(machine_ops):                          # (5): unordered pairs in a fixed total order
        ops_m = machine_ops[machine]
        for first, second in itertools.combinations(ops_m, 2):
            y_index[(first, second, machine)] = n
            n += 1
    for order in range(problem.n_orders):
        z_index[order] = n
        n += 1
    for op in problem.ops:
        c_index[op] = n
        n += 1
    objective = np.zeros(n)
    for order, index in z_index.items():
        objective[index] = 1.0                                    # (1): sum_s z_s (eta = that / S)
    lower = np.zeros(n)
    upper = np.ones(n)
    integrality = np.ones(n, dtype=np.int64)
    for op, index in c_index.items():
        upper[index] = problem.c_ub                               # (9) with the explicit completion bound
        integrality[index] = 0
    model = Model(n, objective, [], lower, upper, integrality, x_index, y_index, z_index, c_index)
    big_l = problem.big_l
    inf = np.inf
    for op in problem.ops:                                        # (2) assignment
        coefficients = {x_index[(op[0], op[1], m)]: 1.0 for m in problem.eligible[op]}
        if variant == "B":
            coefficients[z_index[op[0]]] = -1.0
            model.add_row(coefficients, 0.0, 0.0)
        else:
            model.add_row(coefficients, 1.0, 1.0)
    for op in problem.ops:                                        # (3) arrival and (4) precedence
        order, stage = op
        coefficients = {c_index[op]: 1.0}
        for m in problem.eligible[op]:
            coefficients[x_index[(order, stage, m)]] = -problem.ptime[(order, stage, m)]
        if stage == 0:
            model.add_row(coefficients, problem.arrival(order), inf)
        else:
            coefficients[c_index[(order, stage - 1)]] = -1.0
            model.add_row(coefficients, 0.0, inf)
    for (first, second, machine), y in y_index.items():           # (6) and (7) disjunctive machine constraints
        c1, c2 = c_index[first], c_index[second]
        x1, x2 = x_index[(first[0], first[1], machine)], x_index[(second[0], second[1], machine)]
        p1 = problem.ptime[(first[0], first[1], machine)]
        p2 = problem.ptime[(second[0], second[1], machine)]
        # (6)  C2 >= C1 + p2 - L(1 - y) - L(2 - x1 - x2)   <=>   C2 - C1 - L y - L x1 - L x2 >= p2 - 3L
        model.add_row({c2: 1.0, c1: -1.0, y: -big_l, x1: -big_l, x2: -big_l}, p2 - 3.0 * big_l, inf)
        # (7)  C1 >= C2 + p1 - L y - L(2 - x1 - x2)         <=>   C1 - C2 + L y - L x1 - L x2 >= p1 - 2L
        model.add_row({c1: 1.0, c2: -1.0, y: big_l, x1: -big_l, x2: -big_l}, p1 - 2.0 * big_l, inf)
    for order in range(problem.n_orders):                         # (8) due-date indicator
        last = c_index[(order, problem.stages - 1)]
        model.add_row({last: 1.0, z_index[order]: big_l}, -inf, problem.due(order) + big_l)
    return model


def _sparse(model: Model):
    from scipy.sparse import coo_matrix

    data, rows, cols, lb, ub = [], [], [], [], []
    for row_index, (coefficients, low, high) in enumerate(model.rows):
        for col, value in coefficients.items():
            rows.append(row_index)
            cols.append(col)
            data.append(value)
        lb.append(low)
        ub.append(high)
    matrix = coo_matrix((data, (rows, cols)), shape=(len(model.rows), model.n_vars)).tocsr()
    return matrix, np.asarray(lb), np.asarray(ub)


def solve_scipy(model: Model, time_limit: float) -> dict:
    from scipy.optimize import Bounds, LinearConstraint, milp

    matrix, lb, ub = _sparse(model)
    options = {"disp": False, "mip_rel_gap": 0.0}
    if time_limit and time_limit > 0:
        options["time_limit"] = float(time_limit)
    started = time.perf_counter()
    result = milp(-model.objective, constraints=LinearConstraint(matrix, lb, ub), integrality=model.integrality,
                  bounds=Bounds(model.lower, model.upper), options=options)
    elapsed = time.perf_counter() - started
    status = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded"}.get(int(result.status), "unknown")
    if result.x is None:
        return {"status": status, "objective": float("nan"), "gap": float("nan"), "solve_s": elapsed,
                "values": None, "backend": "highs"}
    gap = float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else 0.0
    return {"status": status, "objective": float(-result.fun), "gap": gap, "solve_s": elapsed,
            "values": np.asarray(result.x), "backend": "highs"}


def solve_gurobi(model: Model, time_limit: float, threads: int = 1) -> dict:
    import gurobipy as gp                      # pragma: no cover - exercised where a licence is available
    from gurobipy import GRB

    matrix, lb, ub = _sparse(model)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("dffsp_milp", env=env)
    vtype = np.where(model.integrality > 0, GRB.BINARY, GRB.CONTINUOUS)
    v = m.addMVar(model.n_vars, lb=model.lower, ub=model.upper, vtype=vtype)
    finite_lb = np.isfinite(lb)
    finite_ub = np.isfinite(ub)
    if finite_lb.any():
        m.addMConstr(matrix[finite_lb], v, ">", lb[finite_lb])
    if finite_ub.any():
        m.addMConstr(matrix[finite_ub], v, "<", ub[finite_ub])
    m.setObjective(model.objective @ v, GRB.MAXIMIZE)
    m.setParam("MIPGap", 0.0)
    m.setParam("Threads", int(threads))
    if time_limit and time_limit > 0:
        m.setParam("TimeLimit", float(time_limit))
    started = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter() - started
    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INFEASIBLE: "infeasible",
              GRB.UNBOUNDED: "unbounded"}.get(m.Status, f"gurobi_{m.Status}")
    if m.SolCount == 0:
        return {"status": status, "objective": float("nan"), "gap": float("nan"), "solve_s": elapsed,
                "values": None, "backend": "gurobi"}
    return {"status": status, "objective": float(m.ObjVal), "gap": float(m.MIPGap), "solve_s": elapsed,
            "values": np.asarray(v.X), "backend": "gurobi"}


def resolve_backend(name: str) -> str:
    """``highs`` (default; ``auto`` and ``scipy`` are aliases) or ``gurobi`` (explicit, unsupported)."""
    name = (name or "highs").lower()
    if name in ("highs", "scipy", "auto"):
        return "highs"
    if name == "gurobi":
        return "gurobi"
    raise ValueError("backend must be highs (auto / scipy) or gurobi")


def solve(model: Model, backend: str, time_limit: float, threads: int = 1) -> dict:
    if backend == "gurobi":
        return solve_gurobi(model, time_limit, threads)
    return solve_scipy(model, time_limit)


# -- solutions ----------------------------------------------------------------------------------------------------
@dataclass
class Schedule:
    """Machine assignment and machine sequences of a solution (what the verifier and the replay consume)."""
    assignment: Dict[Tuple[int, int], int]                  # op -> machine (processed ops only)
    sequences: Dict[int, List[Tuple[int, int]]]             # machine -> ops in processing order
    completion: Dict[Tuple[int, int], float]                # the solver's C_sj (processed ops)
    on_time: List[int]                                      # orders with z_s = 1 (solver)

    @property
    def processed_orders(self) -> set:
        return {op[0] for op in self.assignment}


def extract_schedule(problem: Problem, model: Model, values: np.ndarray) -> Schedule:
    assignment, completion = {}, {}
    for (order, stage, machine), index in model.x_index.items():
        if values[index] > 0.5:
            assignment[(order, stage)] = machine
            completion[(order, stage)] = float(values[model.c_index[(order, stage)]])
    sequences: Dict[int, List[Tuple[int, int]]] = {}
    for op, machine in assignment.items():
        sequences.setdefault(machine, []).append(op)
    for machine, ops in sequences.items():
        p = {op: problem.ptime[(op[0], op[1], machine)] for op in ops}
        ops.sort(key=lambda op: (completion[op] - p[op], completion[op], op))   # by start time
    on_time = [order for order, index in model.z_index.items() if values[index] > 0.5]
    return Schedule(assignment, sequences, completion, sorted(on_time))


def left_shift(problem: Problem, schedule: Schedule) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Re-time the assignment and sequences from scratch: start = max(ready, machine free). Returns
    op -> (start, completion) for the processed ops; raises on an infeasible (cyclic) sequence set."""
    pending = {machine: list(ops) for machine, ops in schedule.sequences.items()}
    machine_free = {machine: 0.0 for machine in pending}
    times: Dict[Tuple[int, int], Tuple[float, float]] = {}
    remaining = sum(len(ops) for ops in pending.values())
    while remaining:
        progressed = False
        for machine, ops in pending.items():
            if not ops:
                continue
            order, stage = ops[0]
            if stage == 0:
                ready = problem.arrival(order)
            elif (order, stage - 1) in times:
                ready = times[(order, stage - 1)][1]
            elif (order, stage - 1) in schedule.assignment:
                continue                                        # predecessor not timed yet
            else:
                raise ValueError(f"operation {(order, stage)} is processed but its predecessor is not")
            start = max(ready, machine_free[machine])
            finish = start + problem.ptime[(order, stage, machine)]
            times[(order, stage)] = (start, finish)
            machine_free[machine] = finish
            ops.pop(0)
            remaining -= 1
            progressed = True
        if not progressed:
            raise ValueError("machine sequences are cyclic: no operation can start")
    return times


def on_time_count(problem: Problem, times: Dict[Tuple[int, int], Tuple[float, float]]) -> int:
    last = problem.stages - 1
    return sum(1 for order in range(problem.n_orders)
               if (order, last) in times and times[(order, last)][1] <= problem.due(order) + EPS)


def verify(problem: Problem, schedule: Schedule, objective: float) -> dict:
    """Left-shift the solution and check the solver's claims: every processed op is timed, the solver's
    completions are feasible (not earlier than the left-shifted ones), z_s = 1 orders are on time."""
    times = left_shift(problem, schedule)
    last = problem.stages - 1
    for op, (start, finish) in times.items():
        if schedule.completion[op] < finish - 1e-4:
            raise AssertionError(f"solver completion {schedule.completion[op]:.3f} of {op} precedes the "
                                 f"left-shifted completion {finish:.3f}")
    for order in schedule.on_time:
        if (order, last) not in times or times[(order, last)][1] > problem.due(order) + EPS:
            raise AssertionError(f"order {order} is claimed on time but is not")
    count = on_time_count(problem, times)
    if count + EPS < round(objective):
        raise AssertionError(f"left-shifted schedule completes {count} orders on time, solver claims {objective}")
    return {"times": times, "on_time": count, "eta": count / max(problem.n_orders, 1)}


# -- simulator replay -------------------------------------------------------------------------------------------
class ScheduleFollowingSimulator(TrainingSimulator):
    """The training simulator restricted to one machine assignment and one sequence per machine: the only
    eligible pair of an idle machine is the next pending operation of its sequence, once that order is waiting
    at that stage. Every other rule of the simulator (arrivals, discards, releases, rewards) is untouched."""

    def __init__(self, spec: InstanceSpec, schedule: Schedule, train_paras: Optional[dict] = None):
        self.schedule = schedule                                # before super().__init__, which calls reset()
        self._pointer: Dict[int, int] = {}
        self.replay_starts: Dict[Tuple[int, int], float] = {}
        super().__init__(spec, dict(REPLAY_PARAS, **(train_paras or {})))

    def reset(self):
        self._pointer = {machine: 0 for machine in self.schedule.sequences}
        self.replay_starts = {}
        return super().reset()

    def _next_pending(self, machine: int):
        ops = self.schedule.sequences.get(machine, [])
        pointer = self._pointer.get(machine, 0)
        while pointer < len(ops):
            order, stage = ops[pointer]
            status = self.order_status[order]
            if status in (self.STATUS_COMPLETED, self.STATUS_DISCARDED) or int(self.order_stage[order]) > stage:
                pointer += 1                                    # gone or already processed elsewhere
                continue
            self._pointer[machine] = pointer
            return order, stage
        self._pointer[machine] = pointer
        return None

    def _eligible_pairs(self, grouped_orders):
        pairs = []
        for machine in range(self.spec.machine_count):
            if self.machine_order[machine] >= 0:
                continue
            pending = self._next_pending(machine)
            if pending is None:
                continue
            order, stage = pending
            if self.order_status[order] != self.STATUS_WAITING or int(self.order_stage[order]) != stage:
                continue
            task = self._task_index(int(self.spec.product_types[order]), stage)
            if order in grouped_orders.get(task, []):
                pairs.append((int(machine), int(task)))
        return pairs

    def act(self):
        """The simulator's action for the first eligible pair, or None when the replay is finished."""
        grouped = self._waiting_orders_by_task()
        pairs = self._eligible_pairs(grouped)
        if not pairs:
            return None
        machine, task = pairs[0]
        order, stage = self._next_pending(machine)
        rank = grouped[task].index(order)
        self.replay_starts[(order, stage)] = float(self.current_time)
        return machine * self.spec.task_count + task, rank


def replay(problem: Problem, schedule: Schedule) -> dict:
    simulator = ScheduleFollowingSimulator(problem.spec, schedule).reset()
    decisions = 0
    while not simulator.done:
        action = simulator.act()
        if action is None:
            simulator._advance_until_decision_or_done()
            if not simulator.done and simulator.act() is None:
                raise AssertionError("replay stalled: a waiting order has no pending operation")
            continue
        simulator.step(action)
        decisions += 1
    last = problem.stages - 1
    on_time = [order for order in range(problem.n_orders)
               if simulator.order_status[order] == simulator.STATUS_COMPLETED]
    return {"eta": simulator.fulfillment_rate, "completed": int(simulator.completed_count),
            "discarded": int(simulator.discarded_count), "decisions": decisions, "on_time": on_time,
            "starts": dict(simulator.replay_starts), "last_stage": last}


# -- brute force --------------------------------------------------------------------------------------------------
def _stage_schedules(problem: Problem, stage: int, orders: Sequence[int]):
    """Every (assignment, machine sequences) of the stage's operations of ``orders``: for each assignment of
    the orders to eligible machines, every sequence of each machine's orders."""
    machines_of = {order: problem.eligible[(order, stage)] for order in orders}
    for assignment in itertools.product(*(machines_of[order] for order in orders)):
        by_machine: Dict[int, List[int]] = {}
        for order, machine in zip(orders, assignment):
            by_machine.setdefault(machine, []).append(order)
        machines = sorted(by_machine)
        for sequences in itertools.product(*(itertools.permutations(by_machine[m]) for m in machines)):
            yield list(zip(machines, sequences))


class BruteForceBudget(Exception):
    """Raised when the enumeration exceeds its node budget."""


def stage_schedule_count(problem: Problem, orders: Sequence[int]) -> int:
    """Upper bound of the sequenced assignments of one stage: n! x C(n + m - 1, m - 1) with m the largest
    eligible machine set of the stage's operations (exact when every order is eligible on the same machines)."""
    from math import comb, factorial

    n = len(orders)
    worst = 0
    for stage in range(problem.stages):
        m = max((len(problem.eligible[(order, stage)]) for order in orders), default=1)
        worst = max(worst, factorial(n) * comb(n + m - 1, m - 1))
    return worst


def brute_force_subset(problem: Problem, orders: Sequence[int], best_so_far: int, counter: list,
                       max_nodes: int = BRUTE_FORCE_MAX_NODES) -> int:
    """Maximum on-time count when exactly ``orders`` are processed (left-shift timing, exhaustive)."""
    orders = list(orders)
    if not orders:
        return 0
    last = problem.stages - 1
    min_remaining = {}                                        # (order, stage) -> minimal processing after this stage
    for order in orders:
        total = 0.0
        for stage in range(last, -1, -1):
            min_remaining[(order, stage)] = total
            total += min(problem.ptime[(order, stage, m)] for m in problem.eligible[(order, stage)])
    best = [best_so_far]

    def recurse(stage: int, ready: Dict[int, float]) -> None:
        counter[0] += 1
        if counter[0] > max_nodes:
            raise BruteForceBudget(f"brute force abandoned after {max_nodes} nodes")
        if stage > last:
            count = sum(1 for order in orders if ready[order] <= problem.due(order) + EPS)
            best[0] = max(best[0], count)
            return
        possible = sum(1 for order in orders
                       if ready[order] + min_remaining[(order, stage)]
                       + min(problem.ptime[(order, stage, m)] for m in problem.eligible[(order, stage)])
                       <= problem.due(order) + EPS)
        if possible <= best[0]:
            return                                             # cannot beat the incumbent
        for stage_schedule in _stage_schedules(problem, stage, orders):
            finish = dict(ready)
            for machine, sequence in stage_schedule:
                free = 0.0
                for order in sequence:
                    start = max(ready[order], free)
                    free = start + problem.ptime[(order, stage, machine)]
                    finish[order] = free
            recurse(stage + 1, finish)

    recurse(0, {order: problem.arrival(order) for order in orders})
    return best[0]


def brute_force(problem: Problem, variant: str, max_stage_schedules: int = BRUTE_FORCE_MAX_STAGE_SCHEDULES,
                max_nodes: int = BRUTE_FORCE_MAX_NODES) -> dict:
    """Optimum of the variant by enumeration: A = all orders processed; B = best over the processed subsets.
    ``status``: ok | skipped_size (a stage has more than ``max_stage_schedules`` sequenced assignments) |
    budget (more than ``max_nodes`` search nodes)."""
    started = time.perf_counter()
    counter = [0]
    orders = list(range(problem.n_orders))
    size = stage_schedule_count(problem, orders)
    if size > max_stage_schedules:
        return {"on_time": None, "eta": None, "nodes": 0, "seconds": 0.0, "status": "skipped_size",
                "stage_schedules": size}
    try:
        if variant == "A":
            best = brute_force_subset(problem, orders, 0, counter, max_nodes)
        else:
            best = 0
            for count in range(len(orders), 0, -1):            # larger subsets first (tighter incumbent)
                if count <= best:
                    break
                for subset in itertools.combinations(orders, count):
                    best = max(best, brute_force_subset(problem, subset, best, counter, max_nodes))
                    if best == count:
                        break
    except BruteForceBudget:
        return {"on_time": None, "eta": None, "nodes": counter[0], "seconds": time.perf_counter() - started,
                "status": "budget", "stage_schedules": size}
    return {"on_time": best, "eta": best / max(problem.n_orders, 1), "nodes": counter[0],
            "seconds": time.perf_counter() - started, "status": "ok", "stage_schedules": size}


# -- driver ------------------------------------------------------------------------------------------------------
def solve_instance(path: Path, variants: Sequence[str], backend: str, time_limit: float,
                   brute_force_max_orders: int, tier: str = "", threads: int = 1, verbose: bool = True,
                   brute_force_max_stage_schedules: int = BRUTE_FORCE_MAX_STAGE_SCHEDULES,
                   brute_force_max_nodes: int = BRUTE_FORCE_MAX_NODES) -> List[dict]:
    spec = parse_instance_file(path)
    problem = build_problem(spec)
    rows = []
    for variant in variants:
        model = build_model(problem, variant)
        result = solve(model, backend, time_limit, threads)
        row = {"instance": path.stem, "tier": tier, "S": problem.n_orders, "n_ops": len(problem.ops),
               "n_binary": model.n_binary, "n_pairs": model.n_pairs, "variant": variant, "completed_opt": "",
               "eta_opt": "", "status": result["status"], "gap": result["gap"], "solve_s": round(result["solve_s"], 3),
               "backend": result["backend"], "big_L": problem.big_l, "c_ub": problem.c_ub,
               "time_limit_s": time_limit, "verifier_eta": "", "replay_eta": "", "replay_ok": "",
               "brute_force_eta": "", "brute_force_ok": "", "brute_force_status": "", "brute_force_nodes": "",
               "brute_force_s": ""}
        if result["values"] is not None:
            objective = result["objective"]
            completed = int(round(objective))
            row["completed_opt"] = completed
            row["eta_opt"] = completed / max(problem.n_orders, 1)
            schedule = extract_schedule(problem, model, result["values"])
            verified = verify(problem, schedule, objective)
            row["verifier_eta"] = verified["eta"]
            replayed = replay(problem, schedule)
            row["replay_eta"] = replayed["eta"]
            ok = (abs(replayed["eta"] - row["eta_opt"]) < EPS if variant == "B"
                  else replayed["eta"] >= row["eta_opt"] - EPS)
            row["replay_ok"] = bool(ok)
            if result["status"] == "optimal" and problem.n_orders <= brute_force_max_orders:
                exhaustive = brute_force(problem, variant, brute_force_max_stage_schedules, brute_force_max_nodes)
                row["brute_force_status"] = exhaustive["status"]
                row["brute_force_nodes"] = exhaustive["nodes"]
                row["brute_force_s"] = round(exhaustive["seconds"], 3)
                if exhaustive["status"] == "ok":
                    row["brute_force_eta"] = exhaustive["eta"]
                    row["brute_force_ok"] = bool(exhaustive["on_time"] == completed)
                    if not row["brute_force_ok"]:
                        row["brute_force_status"] = "mismatch"
        rows.append(row)
        if verbose:
            print(f"[MILP] {path.stem:22s} MILP-{variant} S={problem.n_orders:3d} |K|={len(problem.ops):3d} "
                  f"bin={model.n_binary:5d} {result['status']:10s} eta={row['eta_opt'] if row['eta_opt'] != '' else 'nan'} "
                  f"gap={row['gap']} {result['solve_s']:.1f}s replay_ok={row['replay_ok']} "
                  f"brute={row['brute_force_status'] or '-'}", flush=True)
    return rows


def write_results(rows: List[dict], out_path: Path) -> Path:
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLS).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", nargs="*", default=None, help="instance ids (default: --tiers)")
    parser.add_argument("--tiers", nargs="*", default=list(DEFAULT_TIERS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--backend", default="highs", help="highs (default; auto / scipy are aliases) | gurobi (unsupported)")
    parser.add_argument("--time-limit", type=float, default=600.0, help="seconds per solve (0 = none)")
    parser.add_argument("--threads", type=int, default=1, help="solver threads (gurobi backend only)")
    parser.add_argument("--brute-force-max-orders", type=int, default=6,
                        help="exhaustive cross-check on instances with at most this many orders (0 = off)")
    parser.add_argument("--brute-force-max-stage-schedules", type=int, default=BRUTE_FORCE_MAX_STAGE_SCHEDULES,
                        help="skip the enumeration when a stage has more sequenced assignments than this")
    parser.add_argument("--brute-force-max-nodes", type=int, default=BRUTE_FORCE_MAX_NODES,
                        help="abandon the enumeration after this many search nodes")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    index = load_index()
    if args.instances:
        files = instance_files_by_ids(args.instances)
    else:
        files = []
        for tier in args.tiers:
            files.extend(tier_instance_files(tier))
    if not files:
        raise SystemExit("[FAIL] no instances: run scripts/run_01_prepare_data.py first")
    backend = resolve_backend(args.backend)
    print(f"[MILP] {len(files)} instances, variants {args.variants}, backend {backend}, "
          f"time limit {args.time_limit}s, brute force up to S={args.brute_force_max_orders}")
    rows = []
    started = time.perf_counter()
    for path in files:
        tier = str(index.get(path.stem, {}).get("tier", ""))
        rows.extend(solve_instance(path, args.variants, backend, args.time_limit, args.brute_force_max_orders,
                                   tier=tier, threads=args.threads,
                                   brute_force_max_stage_schedules=args.brute_force_max_stage_schedules,
                                   brute_force_max_nodes=args.brute_force_max_nodes))
    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    write_results(rows, out_path)
    optimal = sum(1 for row in rows if row["status"] == "optimal")
    replay_ok = sum(1 for row in rows if row["replay_ok"] is True)
    brute = [row for row in rows if row["brute_force_ok"] != ""]
    skipped = sum(1 for row in rows if row["brute_force_status"] in ("skipped_size", "budget"))
    print(f"[MILP] {len(rows)} solves: {optimal} optimal, replay consistent {replay_ok}/{len(rows)}, "
          f"brute force consistent {sum(1 for r in brute if r['brute_force_ok'])}/{len(brute)} "
          f"({skipped} skipped by size / budget), {(time.perf_counter() - started) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
