"""Annual liability cash-flow projection, vectorized over scenarios x policies.

Cash flows are net liability outflows at the end of each projection year
(benefits + expenses; single-premium business has no premium income).
"""

import numpy as np

from . import mortality
from .data import SC_SCHEDULE, surrender_charge


def project_deferred(df, a, rates):
    """Deferred fixed annuities. Returns cash flows of shape (n_scenarios, horizon)."""
    H = a["projection_years"]
    m, pb, co = a["mortality"], a["policyholder_behavior"], a["company"]
    earned, new_money = rates["earned"][:, :H], rates["new_money"][:, :H]
    S, P = earned.shape[0], len(df)

    age0 = (df["issue_age"] + df["duration"]).to_numpy()
    dur0 = df["duration"].to_numpy()
    female = (df["gender"] == "F").to_numpy()
    gmir = df["guaranteed_rate"].to_numpy()

    av = np.tile(df["account_value"].to_numpy(dtype=float), (S, 1))
    inforce = np.ones((S, P))
    cf = np.zeros((S, H))
    L = len(SC_SCHEDULE)

    for t in range(H):
        age, dur = age0 + t, dur0 + t
        qx = mortality.q(age, female, t, m, margin=m["margin_deferred"])
        credited = np.maximum(gmir, earned[:, t:t + 1] - co["crediting_spread"])
        competitor = new_money[:, t:t + 1] - co["crediting_spread"]

        base = np.where(dur < L, pb["base_lapse_in_sc"],
                        np.where(dur == L, pb["shock_lapse"], pb["base_lapse_after_sc"]))
        dyn = np.clip(1 + pb["dynamic_lapse_slope"] * (competitor - credited - pb["dynamic_lapse_threshold"]),
                      pb["dynamic_lapse_min"], pb["dynamic_lapse_max"])
        w = np.clip(base * dyn * (1 + pb["lapse_margin"]), 0.0, 1.0)

        av_end = av * (1 + credited)
        deaths = inforce * qx
        lapses = (inforce - deaths) * w
        expense = inforce * co["maintenance_expense"] * (1 + co["expense_inflation"]) ** t

        flows = deaths * av_end + lapses * av_end * (1 - surrender_charge(dur)) + expense
        inforce = inforce - deaths - lapses

        # Remaining contracts are paid out at age 100 or at the end of the horizon
        mature = (age + 1 >= 100) | (t == H - 1)
        flows = flows + np.where(mature, inforce * av_end, 0.0)
        inforce = np.where(mature, 0.0, inforce)

        cf[:, t] = flows.sum(axis=1)
        av = av_end
    return cf


def project_payout(df, a, n_scenarios):
    """Payout annuities. Mortality is deterministic, so cash flows repeat across scenarios."""
    H = a["projection_years"]
    m, co = a["mortality"], a["company"]
    age0 = df["age"].to_numpy()
    female = (df["gender"] == "F").to_numpy()
    pay = df["annual_payment"].to_numpy(dtype=float)
    certain = df["certain_years_remaining"].to_numpy()

    alive = np.ones(len(df))
    cf = np.zeros(H)
    for t in range(H):
        qx = mortality.q(age0 + t, female, t, m, margin=m["margin_payout"])
        alive_end = alive * (1 - qx)
        paid = np.where(t < certain, 1.0, alive_end)
        expense = alive * co["payout_expense"] * (1 + co["expense_inflation"]) ** t
        cf[t] = (pay * paid + expense).sum()
        alive = alive_end
    return np.tile(cf, (n_scenarios, 1))


def discount_factors(earned):
    """End-of-year discount factors from path-wise earned rates."""
    return np.cumprod(1.0 / (1.0 + earned), axis=1)
