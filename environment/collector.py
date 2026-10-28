"""Multi-process trajectory collection: simulators and observation building in worker processes, the policy
in the trainer (new code).

Why: ``TrainingSimulator`` and its observation builder are pure Python / NumPy code that runs serially under
the GIL; with 8 environments the observation building alone takes about half of a rollout on a 4-core CPU.
Worker processes own the simulators and execute nothing but simulator code; every policy forward, action
sample and the PPO update stay in the main process, so with the same ``num_envs`` the collected
``RolloutBuffer`` is bit-identical to the in-process path ``rollout.collect_vectorized_rollouts``
(``tests/test_collector.py``, ``python -m tests.equivalence --collector-workers 3``).

Transport: observations cross the process boundary as NumPy arrays serialised with the plain ``pickle``
module over ``Connection.send_bytes`` / ``recv_bytes``. Torch tensors are never sent through multiprocessing
pipes: once torch is imported, multiprocessing's ForkingPickler moves every tensor into shared memory and
passes file descriptors (one segment per tensor, slow, exhausts file descriptors). Workers draw no random
numbers.

Protocol, strict request / response per worker:
``("load", instance_path, train_paras)`` -> ``("ok", [EnvReport, ...])``;
``("step", [(env_index, (a1, a2)), ...])`` -> ``("ok", [EnvReport, ...])``; ``("close",)`` -> the worker exits.
An exception while handling a command is returned as ``("error", type_name, message, traceback)`` and the
worker keeps serving. Per environment the worker performs exactly the call sequence of the in-process
collector: construct -> [obs] -> (step -> [obs])*, where [obs] is built only when the simulator is not done
and has a decision, so the simulator state trajectory (fluid cache included) is identical.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATS_KEYS = ("fulfillment_rate", "discarded_count", "completed_count", "fluid_solve_count", "fluid_solve_seconds",
              "fluid_cache_hit_count", "fluid_fallback_count", "transition_count", "obs_build_seconds", "step_seconds",
              "fluid_mask_fallback_count", "fluid_mask_unavailable_count", "candidate_pairs_before",
              "candidate_pairs_after", "mask_decision_count")
START_METHODS = ("spawn", "fork", "forkserver")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_PROTOCOL = pickle.HIGHEST_PROTOCOL


class CollectorError(RuntimeError):
    """A worker failed (the remote traceback is included) or the pool is no longer usable."""


@dataclass
class EnvReport:
    env_index: int
    reward: Optional[float]      # None for the initial report of a load
    done: bool
    has_decision: bool
    obs: Optional[dict]          # encoded (NumPy) on the wire, decoded (torch) once in the parent
    stats: Optional[dict]        # the STATS_KEYS of the simulator, present when done


# ------------------------------------------------------------------ codec

def encode_obs(obs: dict) -> dict:
    """Observation dict for the wire: tensors become NumPy arrays (dtype preserved), everything else as is."""
    out = {}
    for key, value in obs.items():
        out[key] = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    return out


def decode_obs(wire: dict) -> dict:
    out = {}
    for key, value in wire.items():
        out[key] = torch.from_numpy(value).clone() if isinstance(value, np.ndarray) else value
    return out


def _dumps(obj) -> bytes:
    return pickle.dumps(obj, protocol=_PROTOCOL)


# ------------------------------------------------------------------ worker

def _capture_stats(simulator) -> dict:
    return {key: getattr(simulator, key) for key in STATS_KEYS}


def _report(simulator, env_index: int, reward: Optional[float]) -> EnvReport:
    """Same per-environment order as the in-process loop: done, has_decision, then the observation."""
    done = bool(simulator.done)
    has_decision = (not done) and bool(simulator.has_decision())
    stats = _capture_stats(simulator) if done else None
    obs = encode_obs(simulator.to_policy_obs()) if (not done and has_decision) else None
    return EnvReport(int(env_index), reward, done, has_decision, obs, stats)


def _worker_main(conn, worker_index: int, env_indices: Sequence[int], threads: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # the parent handles KeyboardInterrupt
    try:
        torch.set_num_threads(max(int(threads), 1))
    except Exception:
        pass
    from data.instance_io import parse_instance_file
    from environment.simulator import TrainingSimulator

    specs: dict = {}
    simulators: dict = {}
    while True:
        try:
            payload = conn.recv_bytes()
        except (EOFError, OSError):
            break                                          # the parent is gone
        try:
            command = pickle.loads(payload)
            kind = command[0]
            if kind == "close":
                break
            if kind == "load":
                _, instance_path, train_paras = command
                key = str(instance_path)
                if key not in specs:
                    specs[key] = parse_instance_file(instance_path)
                spec = specs[key]
                reports = []
                for env_index in env_indices:
                    simulators[env_index] = TrainingSimulator(spec, train_paras)
                    reports.append(_report(simulators[env_index], env_index, None))
                reply = ("ok", reports)
            elif kind == "step":
                _, actions = command
                reports = []
                for env_index, env_action in actions:
                    simulator = simulators[env_index]
                    reward = simulator.step(env_action)
                    reports.append(_report(simulator, env_index, float(reward)))
                reply = ("ok", reports)
            else:
                raise ValueError(f"unknown collector command {kind!r}")
        except Exception as exc:                            # report, keep serving
            reply = ("error", type(exc).__name__, str(exc), traceback.format_exc())
        try:
            conn.send_bytes(_dumps(reply))
        except (BrokenPipeError, OSError):
            break
    try:
        conn.close()
    except Exception:
        pass


# -------------------------------------------------------------------- pool

class CollectorPool:
    """Persistent worker processes owning the ``num_envs`` simulators of a training run.

    Environments are assigned round-robin (``env_index % workers``); results are always returned keyed by
    environment index, so the assignment never affects the collected data. Requests are scattered to all
    workers before any reply is awaited and replies are read in a fixed worker order (no deadlock: commands
    are tiny and every worker answers exactly once per command). A dead worker raises ``CollectorError``
    instead of hanging; a remote exception raises ``CollectorError`` with the remote traceback and leaves the
    pool usable. ``close()`` is idempotent.
    """

    def __init__(self, num_envs: int, workers: int, start_method: str = "spawn", threads_per_worker: int = 1):
        if start_method not in START_METHODS:
            raise ValueError(f"start_method must be one of {START_METHODS}, got {start_method!r}")
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        requested = int(workers)
        self.workers = max(1, min(requested, self.num_envs))
        if self.workers != requested:
            print(f"[Collector] {requested} worker(s) requested for {self.num_envs} environment(s); using {self.workers}",
                  flush=True)
        self.start_method = start_method
        self.threads_per_worker = max(int(threads_per_worker), 1)
        self.blocks: List[List[int]] = [[i for i in range(self.num_envs) if i % self.workers == w]
                                        for w in range(self.workers)]
        self._owner = {i: w for w, block in enumerate(self.blocks) for i in block}
        self._processes = []
        self._conns = []
        self._broken = False
        self._closed = False
        self.startup_seconds = 0.0
        started = time.perf_counter()
        context = mp.get_context(start_method)
        saved = {name: os.environ.get(name) for name in THREAD_ENV}
        for name in THREAD_ENV:
            os.environ[name] = str(self.threads_per_worker)
        try:
            for w, block in enumerate(self.blocks):
                parent_conn, child_conn = context.Pipe(duplex=True)
                process = context.Process(target=_worker_main, args=(child_conn, w, block, self.threads_per_worker),
                                          daemon=True, name=f"collector-{w}")
                process.start()
                child_conn.close()
                self._processes.append(process)
                self._conns.append(parent_conn)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.startup_seconds = time.perf_counter() - started

    # -- lifecycle ------------------------------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @property
    def alive(self) -> bool:
        return (not self._closed) and (not self._broken) and all(p.is_alive() for p in self._processes)

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._broken:
            for conn in self._conns:
                try:
                    conn.send_bytes(_dumps(("close",)))
                except Exception:
                    pass
        for process in self._processes:
            process.join(timeout)
            if process.is_alive():
                process.terminate()
                process.join(1.0)
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass

    # -- transport ------------------------------------------------------------------------------------------
    def _check_usable(self) -> None:
        if self._closed:
            raise CollectorError("collector pool is closed")
        if self._broken:
            raise CollectorError("collector pool is broken (a worker died or the transport failed); create a new one")

    def _send(self, worker: int, command) -> None:
        try:
            self._conns[worker].send_bytes(_dumps(command))
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._broken = True
            raise CollectorError(f"collector worker {worker} is unreachable: {exc}") from exc

    def _recv_raw(self, worker: int, what: str):
        """One reply of ``worker``; raises (and marks the pool broken) only on transport failure."""
        conn, process = self._conns[worker], self._processes[worker]
        while True:
            try:
                if conn.poll(1.0):
                    payload = conn.recv_bytes()
                    break
            except (EOFError, OSError) as exc:
                self._broken = True
                raise CollectorError(f"collector worker {worker} closed the connection during '{what}': {exc}") from exc
            if not process.is_alive():
                self._broken = True
                raise CollectorError(f"collector worker {worker} exited with code {process.exitcode} during '{what}'")
        return pickle.loads(payload)

    def _gather(self, targets, what: str) -> dict:
        """Replies of every target worker (always all of them, so the request / reply alternation stays in
        step); a remote error is raised only after all replies were consumed."""
        replies = {w: self._recv_raw(w, what) for w in targets}
        errors = [(w, reply) for w, reply in replies.items() if reply[0] == "error"]
        if errors:
            worker, (_, type_name, message, remote_traceback) = errors[0]
            more = f" ({len(errors)} workers failed)" if len(errors) > 1 else ""
            raise CollectorError(f"collector worker {worker} failed in '{what}': {type_name}: {message}{more}\n"
                                 f"--- remote traceback ---\n{remote_traceback}")
        return {w: reply[1] for w, reply in replies.items()}

    @staticmethod
    def _decode(report: EnvReport) -> EnvReport:
        if report.obs is not None:
            report.obs = decode_obs(report.obs)
        return report

    # -- commands -------------------------------------------------------------------------------------------
    def load(self, instance_path, train_paras: dict) -> List[EnvReport]:
        """Construct fresh simulators of ``instance_path`` in every worker; reports in environment order."""
        self._check_usable()
        command = ("load", str(instance_path), dict(train_paras))
        for w in range(self.workers):
            self._send(w, command)
        reports: Dict[int, EnvReport] = {}
        for w, worker_reports in self._gather(range(self.workers), "load").items():
            for report in worker_reports:
                reports[report.env_index] = self._decode(report)
        return [reports[i] for i in range(self.num_envs)]

    def step(self, actions: Dict[int, tuple]) -> Dict[int, EnvReport]:
        """Step the given environments (``{env_index: (a1, a2)}``) and return their reports by index."""
        self._check_usable()
        per_worker: Dict[int, list] = {}
        for env_index, env_action in actions.items():
            per_worker.setdefault(self._owner[int(env_index)], []).append(
                (int(env_index), (int(env_action[0]), int(env_action[1]))))
        targets = sorted(per_worker)
        for w in targets:
            self._send(w, ("step", per_worker[w]))
        results: Dict[int, EnvReport] = {}
        for w, worker_reports in self._gather(targets, "step").items():
            for report in worker_reports:
                results[report.env_index] = self._decode(report)
        return results
