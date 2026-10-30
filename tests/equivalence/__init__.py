"""Numerical-equivalence harness.

The harness proves that the refactored code reproduces the original
implementation (git commit c030dfa) bit for bit on CPU:

* L1-L5  component probes (`probe.py`, run in both trees, outputs compared)
* L6     end-to-end training run (3 iterations) compared column by column
* L7     AST identity of functions that cannot be exercised here (the original Gurobi path, unsupported)

Run `python -m tests.equivalence` from the repository root.
"""
