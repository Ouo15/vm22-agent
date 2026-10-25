"""Deterministic input and assumption checks. Each finding is a dict:
{"severity": "error" | "warning" | "info", "check": str, "detail": str}
"""

import pandas as pd

from .data import DEFERRED_COLUMNS, PAYOUT_COLUMNS


def _finding(severity, check, detail):
    return {"severity": severity, "check": check, "detail": detail}


def _common(df, name, required):
    out = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [_finding("error", f"{name}: required columns", f"missing {missing}")]
    nulls = df[list(required)].isna().sum()
    for col, n in nulls[nulls > 0].items():
        out.append(_finding("error", f"{name}: missing values", f"{n} null values in '{col}'"))
    dups = df["policy_id"].duplicated().sum()
    if dups:
        out.append(_finding("error", f"{name}: duplicate policy_id", f"{dups} duplicated ids"))
    bad_gender = (~df["gender"].isin(["M", "F"])).sum()
    if bad_gender:
        out.append(_finding("error", f"{name}: gender codes", f"{bad_gender} values not in {{M, F}}"))
    return out


def check_inputs(portfolio):
    findings = []
    d = portfolio.get("deferred")
    if d is not None and len(d):
        f = _common(d, "deferred", DEFERRED_COLUMNS)
        findings += f
        if not any(x["check"].endswith("required columns") for x in f):
            if (d["account_value"] <= 0).any():
                findings.append(_finding("error", "deferred: account value", f"{(d['account_value'] <= 0).sum()} non-positive account values"))
            age = d["issue_age"] + d["duration"]
            if ((d["issue_age"] < 0) | (age > 105)).any():
                findings.append(_finding("error", "deferred: ages", "issue age < 0 or attained age > 105"))
            if (d["duration"] < 0).any():
                findings.append(_finding("error", "deferred: duration", "negative durations"))
            g = d["guaranteed_rate"]
            if ((g < 0) | (g > 0.06)).any():
                findings.append(_finding("warning", "deferred: guaranteed rate", f"{((g < 0) | (g > 0.06)).sum()} rates outside [0%, 6%]"))
            big = d["account_value"] > d["account_value"].median() * 50
            if big.any():
                findings.append(_finding("warning", "deferred: large contracts", f"{big.sum()} account values > 50x the median; confirm they are not data errors"))
            findings.append(_finding("info", "deferred: summary",
                                     f"{len(d)} contracts, total account value {d['account_value'].sum():,.0f}"))
    p = portfolio.get("payout")
    if p is not None and len(p):
        f = _common(p, "payout", PAYOUT_COLUMNS)
        findings += f
        if not any(x["check"].endswith("required columns") for x in f):
            if (p["annual_payment"] <= 0).any():
                findings.append(_finding("error", "payout: payment", f"{(p['annual_payment'] <= 0).sum()} non-positive payments"))
            if ((p["age"] < 0) | (p["age"] > 110)).any():
                findings.append(_finding("error", "payout: ages", "ages outside [0, 110]"))
            if (p["certain_years_remaining"] < 0).any():
                findings.append(_finding("error", "payout: certain period", "negative certain years"))
            findings.append(_finding("info", "payout: summary",
                                     f"{len(p)} contracts, total annual payments {p['annual_payment'].sum():,.0f}"))
    if not findings:
        findings.append(_finding("error", "portfolio", "no in-force data loaded"))
    return findings


def check_assumptions(a, reg, portfolio=None):
    out = []
    m, pb, co, e = a["mortality"], a["policyholder_behavior"], a["company"], a["economic"]

    if not 0 <= m["improvement_rate"] <= 0.03:
        out.append(_finding("warning", "mortality improvement", f"rate {m['improvement_rate']:.2%} outside [0%, 3%]"))
    if m["margin_payout"] <= 0:
        out.append(_finding("warning", "margin: payout mortality",
                            "no mortality margin on payout annuities; a prudent estimate normally reduces mortality for longevity exposure"))
    if m["margin_deferred"] == 0:
        out.append(_finding("info", "margin: deferred mortality",
                            "no mortality margin on deferred annuities; acceptable only if sensitivity shows mortality is not material"))
    if pb["lapse_margin"] == 0:
        out.append(_finding("warning", "margin: lapse", "no lapse margin; document why or add one"))
    else:
        out.append(_finding("info", "margin: lapse direction",
                            f"lapse margin of {pb['lapse_margin']:+.0%} applied; its direction must be confirmed as prudent with a lapse sensitivity"))
    for k in ("base_lapse_in_sc", "shock_lapse", "base_lapse_after_sc"):
        if not 0 <= pb[k] <= 1:
            out.append(_finding("error", f"lapse: {k}", f"{pb[k]} is not a probability"))
    if pb["shock_lapse"] > 0.6:
        out.append(_finding("warning", "lapse: shock", f"shock lapse {pb['shock_lapse']:.0%} is unusually high"))
    if not pb["dynamic_lapse_min"] <= 1 <= pb["dynamic_lapse_max"]:
        out.append(_finding("error", "lapse: dynamic bounds", "dynamic lapse bounds must bracket 1.0"))
    if co["crediting_spread"] < 0:
        out.append(_finding("error", "crediting spread", "negative crediting spread"))
    if e["volatility"] <= 0:
        out.append(_finding("error", "economic scenarios", "zero volatility makes stochastic scenarios meaningless"))
    if e["default_cost"] <= 0:
        out.append(_finding("warning", "asset defaults", "no default cost deducted from earned rates"))

    if portfolio is not None and portfolio.get("deferred") is not None and len(portfolio["deferred"]):
        d = portfolio["deferred"]
        youngest = int((d["issue_age"] + d["duration"]).min())
        if youngest + a["projection_years"] < 100:
            out.append(_finding("warning", "projection horizon",
                                f"youngest contract only reaches age {youngest + a['projection_years']}; run-off is truncated"))

    unverified = [k for k, v in reg.items() if not v.get("verified")]
    if unverified:
        out.append(_finding("warning", "regulatory parameters",
                            f"not yet verified against the Valuation Manual: {', '.join(unverified)}"))
    out.append(_finding("info", "model scope",
                        "stand-in economic scenario generator and synthetic mortality table are used in place of prescribed sources"))
    return out


def summarize(findings):
    s = pd.Series([f["severity"] for f in findings]).value_counts().to_dict()
    return {k: int(s.get(k, 0)) for k in ("error", "warning", "info")}
