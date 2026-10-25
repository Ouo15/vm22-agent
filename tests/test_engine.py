import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from vm22 import checks, data, mortality, reserves, scenarios
from vm22.projection import discount_factors, project_deferred, project_payout
from vm22.regulatory import RegulatoryIndex
from vm22.workspace import ROOT


@pytest.fixture
def a():
    return yaml.safe_load((ROOT / "config/assumptions.yaml").read_text())


@pytest.fixture
def reg():
    return yaml.safe_load((ROOT / "config/regulatory_parameters.yaml").read_text())


@pytest.fixture
def pf():
    return {"deferred": data.synthetic_deferred(200), "payout": data.synthetic_payout(100)}


def flat_rates(rate, horizon, n=1):
    r = np.full((n, horizon), rate)
    return {"short": r, "new_money": r, "earned": r}


def test_mortality_increases_with_age_and_improves(a):
    m = a["mortality"]
    q = mortality.q(np.arange(60, 100), False, 0, m)
    assert np.all(np.diff(q) > 0)
    assert mortality.q(70, False, 10, m) < mortality.q(70, False, 0, m)
    assert mortality.q(70, True, 0, m) < mortality.q(70, False, 0, m)
    assert mortality.q(70, False, 0, m, margin=0.05) == pytest.approx(0.95 * mortality.q(70, False, 0, m))


def test_payout_reserve_matches_annuity_certain_without_mortality(a):
    """With zero mortality and no expenses, the reserve is an annuity-certain."""
    a = copy.deepcopy(a)
    a["mortality"].update(makeham_a=0.0, gompertz_b=0.0)
    a["company"]["payout_expense"] = 0.0
    a["projection_years"] = 20
    df = pd.DataFrame({"policy_id": ["P1"], "gender": ["M"], "age": [65],
                       "annual_payment": [1000.0], "certain_years_remaining": [0]})
    rates = flat_rates(0.04, 20)
    res = reserves.scenario_reserves(project_payout(df, a, 1), rates["earned"])[0]
    expected = 1000 * (1 - 1.04 ** -20) / 0.04
    assert res == pytest.approx(expected, rel=1e-9)


def test_deferred_runoff_without_decrements(a):
    """No deaths, lapses or expenses: the account value is paid at the horizon."""
    a = copy.deepcopy(a)
    a["mortality"].update(makeham_a=0.0, gompertz_b=0.0)
    a["policyholder_behavior"].update(base_lapse_in_sc=0, shock_lapse=0, base_lapse_after_sc=0)
    a["company"]["maintenance_expense"] = 0.0
    a["projection_years"] = 10
    df = pd.DataFrame({"policy_id": ["D1"], "gender": ["F"], "issue_age": [50], "duration": [0],
                       "account_value": [100_000.0], "guaranteed_rate": [0.0]})
    earned, spread = 0.05, a["company"]["crediting_spread"]
    rates = flat_rates(earned, 10)
    res = reserves.scenario_reserves(project_deferred(df, a, rates), rates["earned"])[0]
    assert res == pytest.approx(100_000 * ((1 + earned - spread) / (1 + earned)) ** 10, rel=1e-9)


def test_guaranteed_rate_binds_when_rates_fall(a):
    a = copy.deepcopy(a)
    a["projection_years"] = 5
    df = data.synthetic_deferred(5)
    df["guaranteed_rate"] = 0.03
    low = project_deferred(df, a, flat_rates(0.01, 5))
    assert np.all(low > 0)


def test_scenario_reserve_is_max_cumulative_pv():
    cf = np.array([[100.0, -300.0, 50.0]])
    earned = np.zeros((1, 3))
    assert reserves.scenario_reserves(cf, earned)[0] == pytest.approx(100.0)


def test_cte():
    v = np.arange(1, 101)
    assert reserves.cte(v, 0.70) == pytest.approx(np.mean(np.arange(71, 101)))
    assert reserves.cte(v, 0.0) == pytest.approx(v.mean())


def test_scenarios_shapes_and_floor(a):
    e = a["economic"]
    s = scenarios.stochastic_set(e, 30, 200)
    assert s["earned"].shape == (200, 30)
    assert s["short"].min() >= e["rate_floor"]
    sert, names = scenarios.sert_set(e, 30)
    assert sert["earned"].shape == (len(names), 30) and names[0] == "baseline"
    assert np.allclose(discount_factors(np.zeros((1, 3))), 1.0)


def test_valuation_end_to_end(pf, a, reg):
    run = reserves.run_valuation(pf, a, reg, n_scenarios=100)
    for cat in ("deferred", "payout"):
        r = run["results"][cat]
        assert r["reported_reserve"] == pytest.approx(max(r["modeled_reserve"], r["floor"]))
        assert r["stochastic_reserve"] >= r["scenario_reserve_percentiles"][50]
    assert run["results"]["total"]["reported_reserve"] == pytest.approx(
        sum(run["results"][c]["reported_reserve"] for c in ("deferred", "payout")))
    assert set(run["unverified_parameters"]) == set(reg)


def test_sensitivity_directions(pf, a, reg):
    up_mort = reserves.run_sensitivity(pf, a, reg, "mortality", 0.10, n_scenarios=100)
    assert up_mort["impact"]["payout"]["change"] < 0          # more deaths, fewer payments
    lower_rates = reserves.run_sensitivity(pf, a, reg, "interest_rates", -0.01, n_scenarios=100)
    assert lower_rates["impact"]["payout"]["change"] > 0      # lower discount rates, higher reserve


def test_input_checks_catch_errors(pf):
    bad = pf["deferred"].copy()
    bad.loc[0, "account_value"] = -5
    bad.loc[1, "policy_id"] = bad.loc[2, "policy_id"]
    found = {f["check"] for f in checks.check_inputs({"deferred": bad}) if f["severity"] == "error"}
    assert "deferred: account value" in found and "deferred: duplicate policy_id" in found
    missing = checks.check_inputs({"deferred": bad.drop(columns=["guaranteed_rate"])})
    assert missing[0]["severity"] == "error" and "required columns" in missing[0]["check"]


def test_assumption_checks_flag_unverified(a, reg):
    findings = checks.check_assumptions(a, reg)
    assert any(f["check"] == "regulatory parameters" for f in findings)


def test_regulatory_search_ranks_and_filters():
    pages = ["VM-22 Section 6 stochastic exclusion ratio test threshold scenarios VM-22",
             "VM-31 report requirements for the PBR actuarial report VM-31",
             "VM-20 net premium reserve life insurance VM-20"]
    idx = RegulatoryIndex.from_pages([p * 10 for p in pages], "manual.pdf")
    hits = idx.search("stochastic exclusion ratio test")
    assert hits[0]["chapter"] == "VM-22" and hits[0]["page"] == 1
    assert all(h["chapter"] == "VM-31" for h in idx.search("report", chapter="VM-31"))
    assert RegulatoryIndex([]).search("anything") == []
