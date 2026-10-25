"""Synthetic in-force blocks and input schemas."""

import numpy as np
import pandas as pd

# 7-year surrender charge schedule, indexed by completed policy years
SC_SCHEDULE = [0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

DEFERRED_COLUMNS = {
    "policy_id": "str",
    "gender": "M/F",
    "issue_age": "int",
    "duration": "int, completed policy years",
    "account_value": "float, USD",
    "guaranteed_rate": "float, annual minimum credited rate",
}
PAYOUT_COLUMNS = {
    "policy_id": "str",
    "gender": "M/F",
    "age": "int, attained age",
    "annual_payment": "float, USD paid at end of each year",
    "certain_years_remaining": "int, years of guaranteed payments left",
}


def synthetic_deferred(n=1000, seed=1):
    """Single-premium deferred fixed annuities with a 7-year surrender charge."""
    rng = np.random.default_rng(seed)
    issue_age = rng.integers(45, 76, n)
    return pd.DataFrame({
        "policy_id": [f"D{i:05d}" for i in range(n)],
        "gender": rng.choice(["M", "F"], n),
        "issue_age": issue_age,
        "duration": rng.integers(0, 10, n),
        "account_value": np.round(rng.lognormal(np.log(90_000), 0.6, n), 2),
        "guaranteed_rate": rng.choice([0.01, 0.015, 0.02, 0.03], n, p=[0.4, 0.3, 0.2, 0.1]),
    })


def synthetic_payout(n=400, seed=2):
    """Single-premium immediate (payout) annuities, some with a certain period."""
    rng = np.random.default_rng(seed)
    years_paid = rng.integers(0, 11, n)
    certain = rng.choice([0, 10], n, p=[0.6, 0.4])
    return pd.DataFrame({
        "policy_id": [f"P{i:05d}" for i in range(n)],
        "gender": rng.choice(["M", "F"], n),
        "age": rng.integers(60, 81, n) + years_paid,
        "annual_payment": np.round(rng.lognormal(np.log(12_000), 0.5, n), 2),
        "certain_years_remaining": np.maximum(certain - years_paid, 0),
    })


def surrender_charge(duration):
    """Surrender charge rate for an array of completed policy years."""
    d = np.asarray(duration)
    sched = np.array(SC_SCHEDULE + [0.0])
    return sched[np.minimum(d, len(SC_SCHEDULE))]


def cash_surrender_value(deferred):
    return float((deferred["account_value"] * (1 - surrender_charge(deferred["duration"]))).sum())
