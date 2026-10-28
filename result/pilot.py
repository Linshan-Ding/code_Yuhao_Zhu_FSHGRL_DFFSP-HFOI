"""Pilot-gate verdict (v2, generalisation paradigm; new code).

The pilot trains the proposed method once for a reduced number of iterations on the training pool and
compares its best validation evaluation with the best deterministic dispatching rule on the same validation
instances (``data/instances/reference.csv`` written by ``baselines/run_rules.py``). The gate passes when the
mean fulfilment of the best validation epoch is at least the mean of the rule references and the policy
reaches the reference on at least ``min_win_share`` of the validation instances (two thirds by default).
"""
from __future__ import annotations

from typing import Sequence


def generalization_verdict(best_rows: Sequence[dict], reference: dict, min_win_share: float = 2 / 3) -> dict:
    """``best_rows``: rows of the best validation epoch (``instance_id, obj``); ``reference``:
    ``{instance_id: obj_ref}`` (minimised objectives). Returns the verdict record."""
    judged = [(row["instance_id"], float(row["obj"]), float(reference[row["instance_id"]]))
              for row in best_rows if row["instance_id"] in reference]
    if not judged:
        return {"verdict": "FAIL", "reason": "no validation instance has a rule reference", "n_judged": 0,
                "wins": 0, "win_share": 0.0, "policy_mean": None, "reference_mean": None}
    wins = sum(1 for _, obj, ref in judged if obj <= ref + 1e-12)
    policy_mean = 1.0 - sum(obj for _, obj, _ in judged) / len(judged)
    reference_mean = 1.0 - sum(ref for _, _, ref in judged) / len(judged)
    win_share = wins / len(judged)
    passed = policy_mean >= reference_mean - 1e-12 and win_share >= float(min_win_share) - 1e-12
    return {
        "verdict": "PASS" if passed else "FAIL",
        "reason": (f"policy mean fulfilment {policy_mean:.4f} vs best-rule mean {reference_mean:.4f}; "
                   f"reaches the reference on {wins}/{len(judged)} validation instances "
                   f"(required share {float(min_win_share):.3f})"),
        "n_judged": len(judged), "wins": wins, "win_share": win_share,
        "policy_mean": policy_mean, "reference_mean": reference_mean,
    }
