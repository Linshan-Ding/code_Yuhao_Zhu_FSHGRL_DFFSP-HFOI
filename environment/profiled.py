"""Timing wrapper of the training simulator, used by eval.py only.

``ProfiledSimulator`` changes no number: it only wraps the fluid-feature computation with a wall-clock
counter (the base class times LP solves, observation building and stepping already) and names the fluid
backend actually in use, so that eval_results.csv can split the decision time of a schedule into policy,
observation, fluid and simulator shares.
"""
from __future__ import annotations

import time

from environment.simulator import TrainingSimulator


class ProfiledSimulator(TrainingSimulator):
    def __init__(self, spec, train_paras: dict):
        self.fluid_feature_seconds = 0.0
        self.fluid_feature_calls = 0
        super().__init__(spec, train_paras)

    def reset(self):
        self.fluid_feature_seconds = 0.0
        self.fluid_feature_calls = 0
        return super().reset()

    def _fluid_features(self, grouped_orders, valid_pairs):
        started = time.perf_counter()
        try:
            return super()._fluid_features(grouped_orders, valid_pairs)
        finally:
            self.fluid_feature_seconds += time.perf_counter() - started
            self.fluid_feature_calls += 1

    @property
    def fluid_backend(self) -> str:
        """``gurobi`` / ``scipy`` (LP solved), ``heuristic_fallback`` (cached_lp requested but no solver),
        ``heuristic`` or ``off`` (see ``TrainingSimulator.fluid_backend_used``)."""
        return self.fluid_backend_used
