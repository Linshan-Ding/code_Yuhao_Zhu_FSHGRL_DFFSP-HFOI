"""Discrete-event training simulator (moved verbatim from agent/training_simulator.py)."""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from data.instance_io import InstanceSpec, parse_instance_file  # noqa: F401  (re-exported for compatibility)


class TrainingSimulator:
    STATUS_NOT_ARRIVED = 0
    STATUS_WAITING = 1
    STATUS_PROCESSING = 2
    STATUS_COMPLETED = 3
    STATUS_DISCARDED = 4
    FLUID_BACKENDS = ("auto", "gurobi", "scipy", "heuristic")
    _GUROBI_IMPORT_ATTEMPTED = False
    _GUROBI_MODULE = None
    _GUROBI_IMPORT_ERROR = None

    def __init__(self, spec: InstanceSpec, train_paras: dict):
        self.spec = spec
        self.order_top_k = int(train_paras.get("order_top_k", 5))
        self.discard_penalty = float(train_paras.get("discard_penalty", 1.0))
        self.progress_reward_coeff = float(train_paras.get("progress_reward_coeff", 0.0))
        self.fluid_mode = str(train_paras.get("fluid_mode", "heuristic")).lower()
        self.fluid_time_limit = float(train_paras.get("fluid_time_limit", 20))
        self.fluid_threads = int(train_paras.get("fluid_threads", 0))
        # v2 (fluid guidance split into its two uses; see docs/experiment-spec.md):
        #   fluid_features  on | off   the two fluid node features (raw_opes[:, 8:10]); off = zero columns
        #   fluid_mask      bool       candidate (machine, operation-type) pairs restricted to u* > threshold,
        #                              fallback to every eligible pair when nothing is retained (cached_lp only)
        #   fluid_backend   auto | gurobi | scipy | heuristic   LP solver; auto = gurobi when importable, else the
        #                              legacy heuristic fallback (fluid_fallback_count counts the attempts)
        # fluid_mode: off keeps its legacy meaning (features off, mask off, no LP).
        self.fluid_backend_name = str(train_paras.get("fluid_backend", "auto")).lower()
        if self.fluid_backend_name not in self.FLUID_BACKENDS:
            raise ValueError(f"fluid_backend must be one of {self.FLUID_BACKENDS}, got {self.fluid_backend_name!r}")
        self.fluid_features_on = (self.fluid_mode != "off"
                                  and str(train_paras.get("fluid_features", "on")).lower() not in ("off", "false", "0"))
        self.fluid_mask_on = (self.fluid_mode == "cached_lp" and bool(train_paras.get("fluid_mask", False)))
        self.fluid_mask_threshold = float(train_paras.get("fluid_mask_threshold", 1e-5))
        #   fluid_recompute decision | event   when the LP is re-solved: at every decision whose state signature
        #                              is new (v2 default; the state cache still avoids repeated solves) or only
        #                              after an arrival / discard event (legacy; a stale solution between events
        #                              cannot serve the mask because its pair set differs from the eligible set)
        self.fluid_recompute = str(train_paras.get("fluid_recompute", "decision")).lower()
        if self.fluid_recompute not in ("decision", "event"):
            raise ValueError(f"fluid_recompute must be 'decision' or 'event', got {self.fluid_recompute!r}")
        self._fluid_active = self.fluid_mode != "off" and (self.fluid_features_on or self.fluid_mask_on)
        self._backend_resolved = None
        self.reset()

    def reset(self):
        self.current_time = float(self.spec.arrival_times.min()) if self.spec.total_orders else 0.0
        self.order_status = np.full(self.spec.total_orders, self.STATUS_NOT_ARRIVED, dtype=np.int8)
        self.order_stage = np.zeros(self.spec.total_orders, dtype=np.int16)
        self.machine_end_times = np.zeros(self.spec.machine_count, dtype=np.float32)
        self.machine_order = np.full(self.spec.machine_count, -1, dtype=np.int32)
        self.machine_stage = np.full(self.spec.machine_count, -1, dtype=np.int16)
        self.completed_count = 0
        self.discarded_count = 0
        self.transition_count = 0
        self.schedule_count = 0
        self.obs_build_seconds = 0.0
        self.step_seconds = 0.0
        self.fluid_solve_count = 0
        self.fluid_solve_seconds = 0.0
        self.fluid_cache_hit_count = 0
        self.fluid_fallback_count = 0
        self.fluid_mask_fallback_count = 0      # mask retained nothing -> every eligible pair kept
        self.fluid_mask_unavailable_count = 0   # mask requested but no LP solution (heuristic fallback)
        self.candidate_pairs_before = 0         # eligible pairs summed over the decisions of the episode
        self.candidate_pairs_after = 0          # candidate pairs offered to the policy, summed likewise
        self.mask_decision_count = 0            # observations built (decisions)
        self._fluid_cache = {}
        self._last_fluid_key = None
        self._fluid_recompute_required = False
        self._last_fluid_features = (
            np.zeros(self.spec.task_count, dtype=np.float32),
            np.zeros(self.spec.task_count, dtype=np.float32),
            None,
        )
        self._last_completion_rate = 0.0
        self._activate_arrivals()
        self._discard_expired_waiting_orders()
        self._advance_until_decision_or_done()
        return self

    @property
    def done(self) -> bool:
        return self.completed_count + self.discarded_count >= self.spec.total_orders

    @property
    def fulfillment_rate(self) -> float:
        return self.completed_count / max(self.spec.total_orders, 1)

    def _task_index(self, product: int, stage: int) -> int:
        return int(product) * self.spec.stage_count + int(stage)

    def _activate_arrivals(self):
        newly_arrived = (
            (self.order_status == self.STATUS_NOT_ARRIVED)
            & (self.spec.arrival_times <= self.current_time)
        )
        if newly_arrived.any():
            self.order_status[newly_arrived] = self.STATUS_WAITING
            self._fluid_recompute_required = True

    def _release_finished_machines(self):
        busy = self.machine_order >= 0
        finished_machines = np.where(busy & (self.machine_end_times <= self.current_time))[0]
        for machine in finished_machines:
            order_index = int(self.machine_order[machine])
            stage = int(self.machine_stage[machine])
            if self.order_status[order_index] == self.STATUS_PROCESSING:
                next_stage = stage + 1
                if next_stage >= self.spec.stage_count:
                    self.order_status[order_index] = self.STATUS_COMPLETED
                    self.completed_count += 1
                else:
                    self.order_stage[order_index] = next_stage
                    self.order_status[order_index] = self.STATUS_WAITING
            self.machine_order[machine] = -1
            self.machine_stage[machine] = -1
            self.machine_end_times[machine] = self.current_time

    def _discard_order(self, order_index: int):
        if self.order_status[order_index] in {
            self.STATUS_COMPLETED,
            self.STATUS_DISCARDED,
        }:
            return
        self.order_status[order_index] = self.STATUS_DISCARDED
        self.discarded_count += 1
        self._fluid_recompute_required = True
        for machine in np.where(self.machine_order == order_index)[0]:
            self.machine_order[machine] = -1
            self.machine_stage[machine] = -1
            self.machine_end_times[machine] = self.current_time

    def _discard_expired_waiting_orders(self):
        waiting = np.where(self.order_status == self.STATUS_WAITING)[0]
        for order_index in waiting:
            if self.current_time >= float(self.spec.due_dates[order_index]):
                self._discard_order(int(order_index))

    def _next_event_time(self) -> float | None:
        future_arrivals = self.spec.arrival_times[
            (self.order_status == self.STATUS_NOT_ARRIVED)
            & (self.spec.arrival_times > self.current_time)
        ]
        busy_ends = self.machine_end_times[
            (self.machine_order >= 0) & (self.machine_end_times > self.current_time)
        ]
        candidates = []
        if future_arrivals.size:
            candidates.append(float(future_arrivals.min()))
        if busy_ends.size:
            candidates.append(float(busy_ends.min()))
        return min(candidates) if candidates else None

    def _waiting_orders_by_task(self) -> Dict[int, List[int]]:
        grouped = {task_index: [] for task_index in range(self.spec.task_count)}
        waiting = np.where(self.order_status == self.STATUS_WAITING)[0]
        for order_index in waiting:
            product = int(self.spec.product_types[order_index])
            stage = int(self.order_stage[order_index])
            task_index = self._task_index(product, stage)
            grouped[task_index].append(int(order_index))
        for task_index, order_indices in grouped.items():
            order_indices.sort(key=lambda idx: float(self.spec.due_dates[idx]))
            if self.order_top_k > 0:
                grouped[task_index] = order_indices[:self.order_top_k]
        return grouped

    def _eligible_pairs(self, grouped_orders: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        pairs = []
        idle_machines = np.where(self.machine_order < 0)[0]
        for task_index, order_indices in grouped_orders.items():
            if not order_indices:
                continue
            _, stage = self.spec.task_to_product_stage[task_index]
            for machine in self.spec.stage_machines[stage]:
                if machine in idle_machines:
                    pairs.append((int(machine), int(task_index)))
        return pairs

    def has_decision(self) -> bool:
        grouped = self._waiting_orders_by_task()
        return bool(self._eligible_pairs(grouped))

    def _advance_until_decision_or_done(self):
        while not self.done and not self.has_decision():
            next_time = self._next_event_time()
            if next_time is None:
                waiting_or_processing = np.where(
                    (self.order_status == self.STATUS_WAITING)
                    | (self.order_status == self.STATUS_PROCESSING)
                    | (self.order_status == self.STATUS_NOT_ARRIVED)
                )[0]
                for order_index in waiting_or_processing:
                    self._discard_order(int(order_index))
                break
            self.current_time = next_time
            self._activate_arrivals()
            self._release_finished_machines()
            self._discard_expired_waiting_orders()

    @classmethod
    def _load_gurobi(cls):
        if not cls._GUROBI_IMPORT_ATTEMPTED:
            cls._GUROBI_IMPORT_ATTEMPTED = True
            try:
                import gurobipy as gp

                cls._GUROBI_MODULE = gp
            except Exception as exc:
                cls._GUROBI_IMPORT_ERROR = exc
                cls._GUROBI_MODULE = None
        return cls._GUROBI_MODULE

    def _heuristic_fluid_features(self, grouped_orders: Dict[int, List[int]]):
        allocations = np.zeros(self.spec.task_count, dtype=np.float32)
        rates = np.zeros(self.spec.task_count, dtype=np.float32)
        if self.fluid_mode == "off":
            return allocations, rates
        for task_index, (_, stage) in enumerate(self.spec.task_to_product_stage):
            order_count = len(grouped_orders.get(task_index, []))
            stage_machines = self.spec.stage_machines[stage]
            allocations[task_index] = float(len(stage_machines))
            if order_count == 0:
                continue
            rates[task_index] = float(sum(
                float(self.spec.proc_rates[task_index, machine].item())
                for machine in stage_machines
            ) / max(order_count, 1))
        return allocations, rates

    def _fluid_cache_key(self, grouped_orders: Dict[int, List[int]],
                         valid_pairs: List[Tuple[int, int]]):
        waiting_signature = tuple(
            (int(task_index), tuple(int(index) for index in order_indices))
            for task_index, order_indices in sorted(grouped_orders.items())
            if order_indices
        )
        idle_machines = tuple(int(index) for index in np.where(self.machine_order < 0)[0])
        unfinished_count = int(np.sum(
            (self.order_status != self.STATUS_COMPLETED)
            & (self.order_status != self.STATUS_DISCARDED)
        ))
        processing_signature = tuple(
            (int(machine), int(self.machine_stage[machine]))
            for machine in np.where(self.machine_order >= 0)[0]
        )
        return (
            idle_machines,
            waiting_signature,
            tuple(sorted((int(machine), int(task)) for machine, task in valid_pairs)),
            processing_signature,
            unfinished_count,
        )

    def _fallback_fluid_features(self, grouped_orders: Dict[int, List[int]]):
        self.fluid_fallback_count += 1
        allocations, rates = self._heuristic_fluid_features(grouped_orders)
        return allocations, rates, None

    def _zero_fluid_features(self):
        return (np.zeros(self.spec.task_count, dtype=np.float32),
                np.zeros(self.spec.task_count, dtype=np.float32), None)

    # -- LP backends (v2) -----------------------------------------------------------------------------------
    @classmethod
    def _load_scipy_linprog(cls):
        try:
            from scipy.optimize import linprog
        except Exception:
            return None
        return linprog

    def resolve_fluid_backend(self) -> str:
        """``gurobi`` | ``scipy`` | ``heuristic``: the solver actually used for ``fluid_mode: cached_lp``.

        ``auto`` prefers Gurobi and otherwise keeps the legacy behaviour (heuristic fallback, counted in
        ``fluid_fallback_count``); ``scipy`` solves the same LP with HiGHS (``scipy.optimize.linprog``), which
        makes the exact fluid solution, and therefore the mask, available without a Gurobi licence.
        """
        if self._backend_resolved is None:
            requested = self.fluid_backend_name
            if requested in ("auto", "gurobi"):
                name = "gurobi" if self._load_gurobi() is not None else "heuristic"
            elif requested == "scipy":
                name = "scipy" if self._load_scipy_linprog() is not None else "heuristic"
            else:
                name = "heuristic"
            self._backend_resolved = name
        return self._backend_resolved

    @property
    def fluid_backend_used(self) -> str:
        """``off`` | ``heuristic`` | ``gurobi`` | ``scipy`` | ``heuristic_fallback`` (cached_lp without a solver)."""
        if not self._fluid_active:
            return "off"
        if self.fluid_mode != "cached_lp":
            return "heuristic"
        backend = self.resolve_fluid_backend()
        return backend if backend != "heuristic" else "heuristic_fallback"

    def _lp_data(self, active_pairs, grouped_orders):
        machines = sorted({machine for machine, _ in active_pairs})
        tasks = sorted({task_index for _, task_index in active_pairs})
        demand = {task_index: float(max(len(grouped_orders.get(task_index, [])), 1)) for task_index in tasks}
        rates = {(machine, task_index): float(self.spec.proc_rates[task_index, machine].item())
                 for machine, task_index in active_pairs}
        return machines, tasks, demand, rates

    def _solve_lp_gurobi(self, gp, active_pairs, grouped_orders):
        """The original model (max-min normalised processing rate); ``{(machine, task): u*}`` or None."""
        model = gp.Model("cached_fluid_lp")
        model.Params.OutputFlag = 0
        if self.fluid_time_limit > 0:
            model.Params.TimeLimit = self.fluid_time_limit
        if self.fluid_threads > 0:
            model.Params.Threads = self.fluid_threads

        x_vars = {}
        for machine, task_index in active_pairs:
            x_vars[(machine, task_index)] = model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"x_{machine}_{task_index}",
            )
        min_rate = model.addVar(lb=0.0, name="min_rate")
        model.update()

        for machine in sorted({machine for machine, _ in active_pairs}):
            model.addConstr(
                gp.quicksum(
                    variable
                    for (candidate_machine, _), variable in x_vars.items()
                    if candidate_machine == machine
                ) <= 1.0,
                name=f"machine_capacity_{machine}",
            )

        active_tasks = sorted({task_index for _, task_index in active_pairs})
        for task_index in active_tasks:
            demand = float(max(len(grouped_orders.get(task_index, [])), 1))
            service = gp.quicksum(
                float(self.spec.proc_rates[task_index, machine].item()) * variable
                for (machine, candidate_task), variable in x_vars.items()
                if candidate_task == task_index
            )
            model.addConstr(
                service >= min_rate * demand,
                name=f"task_rate_floor_{task_index}",
            )

        model.setObjective(min_rate, gp.GRB.MAXIMIZE)
        model.optimize()
        if int(getattr(model, "SolCount", 0)) <= 0:
            return None
        return {pair: max(float(variable.X), 0.0) for pair, variable in x_vars.items()}

    def _solve_lp_scipy(self, linprog, active_pairs, grouped_orders):
        """Same LP with HiGHS: variables u_p in [0, 1] per active pair plus the minimum normalised rate."""
        machines, tasks, demand, rates = self._lp_data(active_pairs, grouped_orders)
        n = len(active_pairs)
        cost = np.zeros(n + 1)
        cost[n] = -1.0                                          # maximise the minimum rate
        rows, bounds_rhs = [], []
        for machine in machines:                                # capacity: sum_p u_p <= 1
            row = np.zeros(n + 1)
            for index, (candidate_machine, _) in enumerate(active_pairs):
                if candidate_machine == machine:
                    row[index] = 1.0
            rows.append(row)
            bounds_rhs.append(1.0)
        for task_index in tasks:                                # rate floor: demand * r - sum mu u <= 0
            row = np.zeros(n + 1)
            for index, (machine, candidate_task) in enumerate(active_pairs):
                if candidate_task == task_index:
                    row[index] = -rates[(machine, task_index)]
            row[n] = demand[task_index]
            rows.append(row)
            bounds_rhs.append(0.0)
        options = {"time_limit": float(self.fluid_time_limit)} if self.fluid_time_limit > 0 else None
        result = linprog(cost, A_ub=np.vstack(rows), b_ub=np.asarray(bounds_rhs),
                         bounds=[(0.0, 1.0)] * n + [(0.0, None)], method="highs", options=options)
        if not getattr(result, "success", False) or result.x is None:
            return None
        return {pair: max(float(result.x[index]), 0.0) for index, pair in enumerate(active_pairs)}

    def _solve_cached_lp(self, grouped_orders: Dict[int, List[int]],
                         valid_pairs: List[Tuple[int, int]]):
        active_pairs = [
            (int(machine), int(task_index))
            for machine, task_index in valid_pairs
            if grouped_orders.get(int(task_index))
        ]
        if not active_pairs:
            allocations, rates = self._heuristic_fluid_features(grouped_orders)
            return allocations, rates, None

        backend = self.resolve_fluid_backend()
        if backend == "heuristic":
            return self._fallback_fluid_features(grouped_orders)

        started = time.perf_counter()
        try:
            self.fluid_solve_count += 1
            if backend == "gurobi":
                solution = self._solve_lp_gurobi(self._load_gurobi(), active_pairs, grouped_orders)
            else:
                solution = self._solve_lp_scipy(self._load_scipy_linprog(), active_pairs, grouped_orders)
            self.fluid_solve_seconds += time.perf_counter() - started
        except Exception:
            self.fluid_solve_seconds += time.perf_counter() - started
            return self._fallback_fluid_features(grouped_orders)
        if solution is None:
            return self._fallback_fluid_features(grouped_orders)

        allocations = np.zeros(self.spec.task_count, dtype=np.float32)
        rates = np.zeros(self.spec.task_count, dtype=np.float32)
        for (machine, task_index), value in solution.items():
            allocations[task_index] += value
            rates[task_index] += float(self.spec.proc_rates[task_index, machine].item()) * value
        for task_index in sorted({task_index for _, task_index in active_pairs}):
            rates[task_index] /= float(max(len(grouped_orders.get(task_index, [])), 1))
        return allocations, rates, solution

    def _fluid_features(self, grouped_orders: Dict[int, List[int]],
                        valid_pairs: List[Tuple[int, int]]):
        """``(allocations[task], rates[task], u*)``; ``u*`` maps (machine, task) to the LP allocation and is
        None without an LP solution (heuristic / off)."""
        if not self._fluid_active:
            return self._zero_fluid_features()
        if self.fluid_mode != "cached_lp":
            allocations, rates = self._heuristic_fluid_features(grouped_orders)
            return allocations, rates, None

        if self.fluid_recompute == "event" and not self._fluid_recompute_required:
            return self._last_fluid_features

        key = self._fluid_cache_key(grouped_orders, valid_pairs)
        if key in self._fluid_cache:
            self.fluid_cache_hit_count += 1
            features = self._fluid_cache[key]
        else:
            features = self._solve_cached_lp(grouped_orders, valid_pairs)
            self._fluid_cache[key] = features
        self._last_fluid_key = key
        self._last_fluid_features = features
        self._fluid_recompute_required = False
        return features

    def _apply_fluid_mask(self, valid_pairs: List[Tuple[int, int]], u_star) -> List[Tuple[int, int]]:
        """Candidate pairs offered to the policy: the eligible pairs with a positive fluid allocation
        (``u* > fluid_mask_threshold``); every eligible pair when nothing would be retained (fallback) or when
        no LP solution exists. Order statistics are kept for the log."""
        self.mask_decision_count += 1
        self.candidate_pairs_before += len(valid_pairs)
        candidates = valid_pairs
        if self.fluid_mask_on:
            if u_star is None:
                self.fluid_mask_unavailable_count += 1
            else:
                kept = [pair for pair in valid_pairs if u_star.get(pair, 0.0) > self.fluid_mask_threshold]
                if kept:
                    candidates = kept
                else:
                    self.fluid_mask_fallback_count += 1
        self.candidate_pairs_after += len(candidates)
        return candidates

    def to_policy_obs(self) -> dict:
        start_time = time.perf_counter()
        grouped_orders = self._waiting_orders_by_task()
        valid_pairs = self._eligible_pairs(grouped_orders)
        task_count = self.spec.task_count
        machine_count = self.spec.machine_count
        max_orders = 1
        if grouped_orders:
            max_orders = max([len(value) for value in grouped_orders.values()] + [1])

        due_dates = torch.zeros((task_count, max_orders), dtype=torch.float32)
        order_mask = torch.zeros((task_count, max_orders), dtype=torch.bool)
        order_counts = torch.zeros((task_count,), dtype=torch.long)
        candidate_order_ids: List[List[int]] = [[] for _ in range(task_count)]
        for task_index, order_indices in grouped_orders.items():
            candidate_order_ids[task_index] = order_indices
            order_counts[task_index] = len(order_indices)
            for rank, order_index in enumerate(order_indices[:max_orders]):
                due_dates[task_index, rank] = float(self.spec.due_dates[order_index] - self.current_time)
                order_mask[task_index, rank] = True

        fluid_allocations, fluid_rates, u_star = self._fluid_features(grouped_orders, valid_pairs)
        candidate_pairs = self._apply_fluid_mask(valid_pairs, u_star)
        eligible_all = torch.zeros((task_count, machine_count), dtype=torch.bool)   # node features (unmasked)
        for machine, task_index in valid_pairs:
            eligible_all[task_index, machine] = True
        eligible = eligible_all
        if candidate_pairs is not valid_pairs:
            eligible = torch.zeros((task_count, machine_count), dtype=torch.bool)   # candidate set (masked)
            for machine, task_index in candidate_pairs:
                eligible[task_index, machine] = True

        raw_opes = torch.zeros((task_count, 12), dtype=torch.float32)
        for task_index, (product, stage) in enumerate(self.spec.task_to_product_stage):
            order_indices = grouped_orders.get(task_index, [])
            due_remaining = [
                float(self.spec.due_dates[index] - self.current_time)
                for index in order_indices
            ]
            stage_machines = self.spec.stage_machines[stage]
            raw_opes[task_index, 0] = len(stage_machines)
            raw_opes[task_index, 1] = int(eligible_all[task_index].sum().item())
            if due_remaining:
                raw_opes[task_index, 2] = float(np.mean(due_remaining))
                raw_opes[task_index, 3] = float(np.min(due_remaining))
                raw_opes[task_index, 4] = float(np.max(due_remaining))
                raw_opes[task_index, 5] = float(np.std(due_remaining))
            raw_opes[task_index, 6] = int(np.sum(          # v2: arrived, unfinished orders only (no future orders)
                ((self.order_status == self.STATUS_WAITING) | (self.order_status == self.STATUS_PROCESSING))
                & (self.spec.product_types == product)
                & (self.order_stage <= stage)
            ))
            raw_opes[task_index, 7] = len(order_indices)
            if self.fluid_features_on:
                raw_opes[task_index, 8] = float(fluid_allocations[task_index])
                raw_opes[task_index, 9] = float(fluid_rates[task_index])
            raw_opes[task_index, 10] = product
            raw_opes[task_index, 11] = stage

        raw_mas = torch.zeros((machine_count, 5), dtype=torch.float32)
        for machine in range(machine_count):
            idle = self.machine_order[machine] < 0
            candidate_tasks = [
                task_index
                for task_index, (_, stage) in enumerate(self.spec.task_to_product_stage)
                if machine in self.spec.stage_machines[stage]
            ]
            raw_mas[machine, 0] = len(candidate_tasks)
            raw_mas[machine, 1] = self.current_time if idle else float(self.machine_end_times[machine])
            raw_mas[machine, 2] = 1.0 if idle else 0.0
            raw_mas[machine, 3] = int(sum(eligible_all[task_index, machine].item() for task_index in candidate_tasks))
            raw_mas[machine, 4] = 0.0 if idle else 1.0

        valid_action_tokens = int(sum(
            len(grouped_orders.get(task_index, []))
            for _, task_index in candidate_pairs
        ))
        padded_action_tokens = max(machine_count * task_count * max_orders, 1)
        self.obs_build_seconds += time.perf_counter() - start_time
        return {
            "raw_opes": raw_opes,
            "raw_mas": raw_mas,
            "proc_time": self.spec.proc_rates.clone(),
            "ope_ma_adj": self.spec.ope_ma_adj.clone(),
            "ope_pre_adj": self.spec.ope_pre_adj.clone(),
            "ope_sub_adj": self.spec.ope_sub_adj.clone(),
            "eligible": eligible,
            "due_dates": due_dates,
            "order_mask": order_mask,
            "order_counts": order_counts,
            "candidate_order_ids": candidate_order_ids,
            "valid_pairs": torch.tensor(candidate_pairs, dtype=torch.long) if candidate_pairs else torch.zeros((0, 2), dtype=torch.long),
            "current_order_completed_rate": self.fulfillment_rate,
            "num_tasks": task_count,
            "num_machines": machine_count,
            "max_orders": max_orders,
            "valid_pair_count": len(candidate_pairs),
            "valid_order_count": int(order_mask.sum().item()),
            "valid_action_tokens": valid_action_tokens,
            "padded_action_tokens": padded_action_tokens,
            "order_top_k": self.order_top_k,
        }

    def step(self, env_action: Tuple[int, int]) -> float:
        start_time = time.perf_counter()
        if self.done:
            return 0.0

        previous_completed = self.completed_count
        previous_discarded = self.discarded_count
        previous_rate = self.fulfillment_rate
        grouped_orders = self._waiting_orders_by_task()
        action1, order_rank = int(env_action[0]), int(env_action[1])
        task_count = self.spec.task_count
        machine = action1 // task_count
        task_index = action1 % task_count
        valid_pairs = set(self._eligible_pairs(grouped_orders))
        if (machine, task_index) not in valid_pairs:          # v2: no silent remapping (known issue 12)
            raise ValueError(f"invalid action: (machine {machine}, operation type {task_index}) is not an eligible "
                             f"pair at t={self.current_time:.0f} ({len(valid_pairs)} eligible pair(s))")

        order_indices = grouped_orders.get(task_index, [])
        if not 0 <= order_rank < len(order_indices):
            raise ValueError(f"invalid action: queue rank {order_rank} out of range for operation type {task_index} "
                             f"({len(order_indices)} candidate order(s))")
        order_index = order_indices[order_rank]
        product, stage = self.spec.task_to_product_stage[task_index]
        process_time = float(self.spec.process_times[product, machine])
        finish_time = self.current_time + process_time

        if finish_time > float(self.spec.due_dates[order_index]):
            self._discard_order(order_index)
        else:
            self.order_status[order_index] = self.STATUS_PROCESSING
            self.machine_order[machine] = order_index
            self.machine_stage[machine] = stage
            self.machine_end_times[machine] = finish_time
            self.schedule_count += 1

        self.transition_count += 1
        self._activate_arrivals()
        self._release_finished_machines()
        self._discard_expired_waiting_orders()
        self._advance_until_decision_or_done()
        total_orders = max(self.spec.total_orders, 1)
        reward = (
            (self.completed_count - previous_completed) / total_orders
            - self.discard_penalty * (self.discarded_count - previous_discarded) / total_orders
            + self.progress_reward_coeff * (self.fulfillment_rate - previous_rate)
        )
        self._last_completion_rate = self.fulfillment_rate
        self.step_seconds += time.perf_counter() - start_time
        return reward
