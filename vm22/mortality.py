"""Synthetic annuitant mortality with mortality improvement.

Gompertz-Makeham base table, graded improvement by age. A stand-in for the
prescribed valuation table; replace with a real table for any serious use.
"""

import numpy as np


def base_q(age, female, m):
    """One-year death probability from the base table."""
    age = np.asarray(age, dtype=float)
    mu = m["makeham_a"] + m["gompertz_b"] * m["gompertz_c"] ** age
    q = 1.0 - np.exp(-mu)
    q = np.where(np.asarray(female), q * m["female_factor"], q)
    return np.clip(np.where(age >= 120, 1.0, q), 0.0, 1.0)


def improvement_rate(age, m):
    """Flat improvement to `improvement_grade_from_age`, graded linearly to 0."""
    age = np.asarray(age, dtype=float)
    lo, hi = m["improvement_grade_from_age"], m["improvement_zero_age"]
    grade = np.clip((hi - age) / (hi - lo), 0.0, 1.0)
    return m["improvement_rate"] * grade


def q(age, female, years_from_base, m, margin=0.0):
    """Death probability with improvement for `years_from_base` years and a prudence margin.

    A positive margin reduces mortality (prudent for longevity exposure).
    """
    qx = base_q(age, female, m) * (1.0 - improvement_rate(age, m)) ** years_from_base
    return np.clip(qx * (1.0 - margin), 0.0, 1.0)
