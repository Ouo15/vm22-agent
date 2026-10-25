"""Reserve calculations: scenario reserves, DR, SR (CTE), SERT, floor, sensitivities.

Scenario reserve: the starting asset amount needed so that a portfolio earning
the scenario's net earned rate never runs out while paying the liability cash
flows. Equivalently, the maximum over t of the cumulative present value of net
cash flows (the "greatest present value of accumulated deficiencies" framing).
"""

import copy
import uuid
from datetime import datetime, timezone

import numpy as np

from . import scenarios
from .data import cash_surrender_value
from .projection import discount_factors, project_deferred, project_payout

CATEGORIES = ("deferred", "payout")


def scenario_reserves(cf, earned):
    pv = np.cumsum(cf * discount_factors(earned[:, :cf.shape[1]]), axis=1)
    return pv.max(axis=1)


def cte(values, level):
    """Mean of the worst (1 - level) share of outcomes (highest reserves)."""
    v = np.sort(np.asarray(values))[::-1]
    k = max(1, int(np.ceil(len(v) * (1 - level) - 1e-9)))  # tolerance: 100 * 0.3 is 30.000000000000004
    return float(v[:k].mean())


def _cash_flows(portfolio, a, rates, category):
    if category == "deferred":
        return project_deferred(portfolio["deferred"], a, rates)
    return project_payout(portfolio["payout"], a, rates["earned"].shape[0])


def _floor(portfolio, category, reg):
    if reg["reserve_floor_basis"]["value"] == "cash_surrender_value" and category == "deferred":
        return cash_surrender_value(portfolio["deferred"])
    return 0.0


def run_sert(portfolio, a, reg, rate_shift=0.0):
    """Ratio test per category: (worst stress reserve - baseline reserve) / PV of baseline benefits."""
    H = a["projection_years"]
    rates, names = scenarios.sert_set(a["economic"], H, rate_shift)
    threshold = reg["sert_threshold"]["value"]
    out = {}
    for cat in CATEGORIES:
        if portfolio.get(cat) is None or len(portfolio[cat]) == 0:
            continue
        cf = _cash_flows(portfolio, a, rates, cat)
        res = scenario_reserves(cf, rates["earned"])
        base = res[0]
        pv_benefits = float((cf[0] * discount_factors(rates["earned"][:1, :H])[0]).sum())
        worst = int(np.argmax(res))
        ratio = (res[worst] - base) / pv_benefits
        out[cat] = {
            "ratio": float(ratio),
            "threshold": threshold,
            "passed": bool(ratio < threshold),
            "worst_scenario": names[worst],
            "scenario_reserves": dict(zip(names, map(float, res))),
        }
    return out


def run_valuation(portfolio, a, reg, n_scenarios=None, rate_shift=0.0, sert=None, keep_distribution=True):
    """Full valuation: SERT, DR, SR, floor and reported reserve per category."""
    H = a["projection_years"]
    e = a["economic"]
    n = int(n_scenarios or reg["n_stochastic_scenarios"]["value"])
    level = reg["sr_cte_level"]["value"]
    sert = sert or run_sert(portfolio, a, reg, rate_shift)
    base_rates = scenarios.baseline_set(e, H, rate_shift)
    stoch_rates = scenarios.stochastic_set(e, H, n, rate_shift)

    results = {}
    for cat in CATEGORIES:
        if portfolio.get(cat) is None or len(portfolio[cat]) == 0:
            continue
        base_cf = _cash_flows(portfolio, a, base_rates, cat)
        dr = float(scenario_reserves(base_cf, base_rates["earned"])[0])
        sr_dist = scenario_reserves(_cash_flows(portfolio, a, stoch_rates, cat), stoch_rates["earned"])
        sr = cte(sr_dist, level)
        stochastic_required = not sert[cat]["passed"]
        modeled = sr if stochastic_required else dr
        floor = _floor(portfolio, cat, reg)
        results[cat] = {
            "policies": int(len(portfolio[cat])),
            "deterministic_reserve": dr,
            "stochastic_reserve": sr,
            "stochastic_required": stochastic_required,
            "modeled_reserve": modeled,
            "floor": floor,
            "floor_binding": floor > modeled,
            "reported_reserve": max(modeled, floor),
            "scenario_reserve_percentiles": {
                p: float(np.percentile(sr_dist, p)) for p in (5, 25, 50, 75, 95, 99)
            },
            "baseline_cash_flows": [float(x) for x in base_cf[0]],
        }
        if keep_distribution:
            results[cat]["scenario_reserves"] = [float(x) for x in sr_dist]

    results["total"] = {
        k: sum(results[c][k] for c in CATEGORIES if c in results)
        for k in ("deterministic_reserve", "stochastic_reserve", "modeled_reserve", "floor", "reported_reserve")
    }
    return {
        "run_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_scenarios": n,
        "cte_level": level,
        "rate_shift": rate_shift,
        "sert": sert,
        "results": results,
        "unverified_parameters": sorted(k for k, v in reg.items() if not v.get("verified")),
    }


# Sensitivity shocks: name -> function(assumptions copy, shock) applied in place
SENSITIVITIES = {
    "mortality": lambda a, s: a["mortality"].update(gompertz_b=a["mortality"]["gompertz_b"] * (1 + s),
                                                    makeham_a=a["mortality"]["makeham_a"] * (1 + s)),
    "mortality_improvement": lambda a, s: a["mortality"].update(improvement_rate=a["mortality"]["improvement_rate"] + s),
    "lapse": lambda a, s: a["policyholder_behavior"].update(
        base_lapse_in_sc=a["policyholder_behavior"]["base_lapse_in_sc"] * (1 + s),
        shock_lapse=a["policyholder_behavior"]["shock_lapse"] * (1 + s),
        base_lapse_after_sc=a["policyholder_behavior"]["base_lapse_after_sc"] * (1 + s)),
    "crediting_spread": lambda a, s: a["company"].update(crediting_spread=a["company"]["crediting_spread"] + s),
    "expenses": lambda a, s: a["company"].update(maintenance_expense=a["company"]["maintenance_expense"] * (1 + s),
                                                 payout_expense=a["company"]["payout_expense"] * (1 + s)),
    "interest_rates": None,  # handled as a parallel shift of all rate paths
}

SENSITIVITY_UNITS = {
    "mortality": "relative change in mortality rates (0.10 = +10%)",
    "mortality_improvement": "absolute change in annual improvement rate (0.005 = +0.5%)",
    "lapse": "relative change in lapse rates (0.10 = +10%)",
    "crediting_spread": "absolute change in crediting spread (0.005 = +50bp)",
    "expenses": "relative change in expenses (0.10 = +10%)",
    "interest_rates": "parallel shift in all interest rates (0.01 = +100bp)",
}


def run_sensitivity(portfolio, a, reg, assumption, shock, n_scenarios=None):
    """Re-run the valuation with one assumption shocked; return reserve changes by category."""
    if assumption not in SENSITIVITIES:
        raise ValueError(f"unknown assumption '{assumption}'")
    base = run_valuation(portfolio, a, reg, n_scenarios, keep_distribution=False)
    shocked_a = copy.deepcopy(a)
    rate_shift = 0.0
    if assumption == "interest_rates":
        rate_shift = shock
    else:
        SENSITIVITIES[assumption](shocked_a, shock)
    shocked = run_valuation(portfolio, shocked_a, reg, n_scenarios, rate_shift=rate_shift, keep_distribution=False)

    out = {}
    for cat in list(CATEGORIES) + ["total"]:
        if cat not in base["results"]:
            continue
        b, s = base["results"][cat]["reported_reserve"], shocked["results"][cat]["reported_reserve"]
        out[cat] = {"base_reported": b, "shocked_reported": s, "change": s - b,
                    "change_pct": (s - b) / b if b else None}
        if cat != "total":
            out[cat]["base_modeled"] = base["results"][cat]["modeled_reserve"]
            out[cat]["shocked_modeled"] = shocked["results"][cat]["modeled_reserve"]
    return {"assumption": assumption, "shock": shock, "unit": SENSITIVITY_UNITS[assumption], "impact": out}
