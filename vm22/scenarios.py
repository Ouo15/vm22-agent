"""Interest-rate scenarios: a one-factor Vasicek stand-in for the NAIC GOES.

Each scenario set returns annual arrays of shape (n_scenarios, horizon):
short rate, new-money yield, and the net earned rate of a portfolio that
reinvests gradually (so earned rates lag market rates).
"""

import numpy as np

# Simplified analog of a prescribed deterministic stress set for the exclusion
# test: shifts (in decimal) added to the baseline short-rate path.
SERT_SHIFTS = {
    "baseline": lambda t: 0.0 * t,
    "up_ramp_300": lambda t: 0.03 * np.minimum(t / 10, 1),
    "down_ramp_300": lambda t: -0.03 * np.minimum(t / 10, 1),
    "up_shock_200": lambda t: 0.02 + 0.0 * t,
    "down_shock_200": lambda t: -0.02 + 0.0 * t,
    "up_then_down": lambda t: np.where(t <= 5, 0.02 * t / 5, 0.02 - 0.03 * np.minimum((t - 5) / 5, 1)),
    "down_then_up": lambda t: np.where(t <= 5, -0.02 * t / 5, -0.02 + 0.03 * np.minimum((t - 5) / 5, 1)),
    "pop_up_100": lambda t: np.where(t < 3, 0.01, 0.0),
}


def baseline_short_rate(e, horizon):
    """Expected Vasicek path: reverts from the initial to the long-term rate."""
    t = np.arange(horizon)
    return e["long_term_rate"] + (e["initial_short_rate"] - e["long_term_rate"]) * (1 - e["mean_reversion"]) ** t


def stochastic_short_rates(e, horizon, n, seed=None):
    rng = np.random.default_rng(e["seed"] if seed is None else seed)
    r = np.empty((n, horizon))
    r[:, 0] = e["initial_short_rate"]
    shocks = rng.standard_normal((n, horizon))
    for t in range(1, horizon):
        r[:, t] = r[:, t - 1] + e["mean_reversion"] * (e["long_term_rate"] - r[:, t - 1]) + e["volatility"] * shocks[:, t]
    return np.maximum(r, e["rate_floor"])


def build(short, e, rate_shift=0.0):
    """Derive market and portfolio rates from short-rate paths."""
    short = np.atleast_2d(short) + rate_shift
    new_money = short + e["term_premium_10y"] + e["portfolio_spread"]
    book = np.empty_like(short)
    k = e["reinvestment_turnover"]
    book[:, 0] = e["portfolio_book_yield"] + rate_shift
    for t in range(1, short.shape[1]):
        book[:, t] = (1 - k) * book[:, t - 1] + k * new_money[:, t - 1]
    return {"short": short, "new_money": new_money, "earned": book - e["default_cost"]}


def baseline_set(e, horizon, rate_shift=0.0):
    return build(baseline_short_rate(e, horizon), e, rate_shift)


def stochastic_set(e, horizon, n, rate_shift=0.0):
    return build(stochastic_short_rates(e, horizon, n), e, rate_shift)


def sert_set(e, horizon, rate_shift=0.0):
    t = np.arange(horizon, dtype=float)
    base = baseline_short_rate(e, horizon)
    paths = np.vstack([np.maximum(base + f(t), e["rate_floor"]) for f in SERT_SHIFTS.values()])
    return build(paths, e, rate_shift), list(SERT_SHIFTS)
