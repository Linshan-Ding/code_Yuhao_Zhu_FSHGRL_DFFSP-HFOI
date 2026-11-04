"""Shared fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _every_method_unless_the_test_says_otherwise(monkeypatch):
    """The synthetic frames of the tests include every method: the paper's default exclusion of SPT
    (configs/runs.yaml) is switched off unless a test sets FSHGRL_EXCLUDE_METHODS itself."""
    monkeypatch.setenv("FSHGRL_EXCLUDE_METHODS", "")
