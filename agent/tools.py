"""Tool definitions for the VM-22 agent, input validation, and dispatch to the workspace."""

import json

from vm22 import DISCLAIMER
from vm22.reports import REPORT_KINDS
from vm22.reserves import SENSITIVITY_UNITS


def _tool(name, description, properties=None, required=()):
    return {
        "name": name,
        "description": description,
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": list(required),
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool("search_valuation_manual",
          "Search the loaded NAIC Valuation Manual (or other loaded regulatory PDFs) and return the most "
          "relevant passages with document, chapter and page for citation. Use for any question about VM-22 "
          "or VM-31 requirements and before confirming a regulatory parameter.",
          {"query": {"type": "string", "description": "What to look for, in the manual's own terms."},
           "chapter": {"type": "string", "enum": ["VM-22", "VM-31", "VM-20", "VM-21", "any"],
                       "description": "Restrict to one chapter. Defaults to any."},
           "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Passages to return (default 5)."}},
          ["query"]),
    _tool("describe_portfolio",
          "Summarize the loaded in-force block: contract counts, account values, ages, guarantees, payments, "
          "and whether the data is synthetic or uploaded."),
    _tool("validate_inputs",
          "Run data validation checks on the in-force block (required fields, ranges, duplicates, outliers)."),
    _tool("review_assumptions",
          "Check the assumption set and regulatory parameters: margins, bounds, projection horizon, and "
          "parameters not yet verified against the Valuation Manual. Returns the assumption values too."),
    _tool("run_sert",
          "Run the Stochastic Exclusion Ratio Test per reserving category against a set of deterministic "
          "interest-rate stress scenarios. Shows which categories need a Stochastic Reserve."),
    _tool("run_valuation",
          "Run the full valuation: SERT, Deterministic Reserve, Stochastic Reserve (CTE over stochastic "
          "scenarios), floor and reported reserve by category. Saves a run record.",
          {"n_scenarios": {"type": "integer", "minimum": 50, "maximum": 2000,
                           "description": "Stochastic scenarios; defaults to the regulatory parameter."}}),
    _tool("run_sensitivity",
          "Re-run the valuation with one assumption shocked and report the change in reported and modeled "
          "reserves by category. Units: " + "; ".join(f"{k}: {v}" for k, v in SENSITIVITY_UNITS.items()),
          {"assumption": {"type": "string", "enum": list(SENSITIVITY_UNITS)},
           "shock": {"type": "number", "minimum": -0.5, "maximum": 0.5},
           "n_scenarios": {"type": "integer", "minimum": 50, "maximum": 2000}},
          ["assumption", "shock"]),
    _tool("get_run",
          "Get the summary of a saved valuation run (the latest if run_id is omitted), including the "
          "assumptions and regulatory parameters it used.",
          {"run_id": {"type": "string"}}),
    _tool("confirm_regulatory_parameter",
          "Set a regulatory parameter's value and mark it verified. Only use with a citation taken from a "
          "search_valuation_manual result in this conversation.",
          {"name": {"type": "string", "enum": ["sr_cte_level", "sert_threshold", "n_stochastic_scenarios",
                                               "reserve_floor_basis", "reserve_rule"]},
           "value": {"type": ["number", "string"]},
           "citation": {"type": "string", "description": "Document, chapter, page and quoted text supporting the value."}},
          ["name", "value", "citation"]),
    _tool("save_report",
          "Save a drafted report (Markdown) so the user can review and download it. Kinds: "
          + "; ".join(f"{k}: {v}" for k, v in REPORT_KINDS.items()),
          {"kind": {"type": "string", "enum": list(REPORT_KINDS)},
           "title": {"type": "string"},
           "markdown": {"type": "string", "description": "Full report body in Markdown (no top-level title)."}},
          ["kind", "title", "markdown"]),
]

_SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}
_TYPES = {"string": str, "integer": int, "number": (int, float), "object": dict, "boolean": bool}


def validate(name, args):
    """Validate tool input against its schema subset. Returns an error message or None."""
    schema = _SCHEMAS.get(name)
    if schema is None:
        return f"unknown tool '{name}'"
    if not isinstance(args, dict):
        return "input must be a JSON object"
    props = schema["properties"]
    for key in schema["required"]:
        if key not in args:
            return f"missing required field '{key}'"
    for key, value in args.items():
        if key not in props:
            return f"unexpected field '{key}'"
        spec = props[key]
        types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        if isinstance(value, bool) or not any(isinstance(value, _TYPES[t]) for t in types):
            return f"field '{key}' must be {' or '.join(types)}"
        if "enum" in spec and value not in spec["enum"]:
            return f"field '{key}' must be one of {spec['enum']}"
        if "minimum" in spec and value < spec["minimum"] or "maximum" in spec and value > spec["maximum"]:
            return f"field '{key}' out of range [{spec.get('minimum')}, {spec.get('maximum')}]"
    return None


def _k(x):
    return round(x / 1000.0, 1) if isinstance(x, (int, float)) else x


def _run_summary(run):
    res = {}
    for cat, r in run["results"].items():
        if cat == "total":
            res[cat] = {k: _k(v) for k, v in r.items()}
            continue
        res[cat] = {
            "policies": r["policies"],
            "deterministic_reserve": _k(r["deterministic_reserve"]),
            "stochastic_reserve": _k(r["stochastic_reserve"]),
            "stochastic_required": r["stochastic_required"],
            "modeled_reserve": _k(r["modeled_reserve"]),
            "floor": _k(r["floor"]),
            "floor_binding": r["floor_binding"],
            "reported_reserve": _k(r["reported_reserve"]),
            "scenario_reserve_percentiles": {p: _k(v) for p, v in r["scenario_reserve_percentiles"].items()},
            "baseline_cash_flows_first_10_years": [_k(v) for v in r["baseline_cash_flows"][:10]],
        }
    return {
        "run_id": run["run_id"], "timestamp": run["timestamp"], "units": "USD thousands",
        "n_scenarios": run["n_scenarios"], "cte_level": run["cte_level"],
        "sert": {c: {k: v for k, v in s.items() if k != "scenario_reserves"} for c, s in run["sert"].items()},
        "results": res, "unverified_parameters": run["unverified_parameters"],
        "check_counts": run.get("checks"), "portfolio_source": run.get("portfolio_source"),
        "model_disclaimer": DISCLAIMER,
    }


def _portfolio_summary(ws):
    out = {"source": ws.portfolio_source}
    d, p = ws.portfolio.get("deferred"), ws.portfolio.get("payout")
    if d is not None and len(d):
        age = d["issue_age"] + d["duration"]
        out["deferred"] = {
            "contracts": len(d), "total_account_value_k": _k(d["account_value"].sum()),
            "average_account_value": round(float(d["account_value"].mean())),
            "attained_age_min_mean_max": [int(age.min()), round(float(age.mean()), 1), int(age.max())],
            "share_female": round(float((d["gender"] == "F").mean()), 3),
            "guaranteed_rate_mix": {f"{k:.2%}": int(v) for k, v in d["guaranteed_rate"].value_counts().sort_index().items()},
            "contracts_in_surrender_charge_period": int((d["duration"] < 7).sum()),
            "contracts_reaching_shock_lapse_next_year": int((d["duration"] == 7).sum()),
        }
    if p is not None and len(p):
        out["payout"] = {
            "contracts": len(p), "total_annual_payments_k": _k(p["annual_payment"].sum()),
            "age_min_mean_max": [int(p["age"].min()), round(float(p["age"].mean()), 1), int(p["age"].max())],
            "share_female": round(float((p["gender"] == "F").mean()), 3),
            "contracts_with_certain_period_left": int((p["certain_years_remaining"] > 0).sum()),
        }
    return out


def execute(ws, name, args):
    """Run one tool. Returns (result_text, is_error)."""
    err = validate(name, args)
    if err:
        return json.dumps({"error": err}), True
    try:
        if name == "search_valuation_manual":
            if not ws.reg_index.records:
                return json.dumps({"error": "No Valuation Manual is loaded. Ask the user to upload the NAIC "
                                            "Valuation Manual PDF in the sidebar."}), True
            hits = ws.reg_index.search(args["query"], args.get("chapter", "any"), args.get("top_k", 5))
            return json.dumps({"results": hits or [], "note": None if hits else "no relevant passages found"}), False
        if name == "describe_portfolio":
            return json.dumps(_portfolio_summary(ws)), False
        if name == "validate_inputs":
            return json.dumps({"findings": ws.input_checks()}), False
        if name == "review_assumptions":
            return json.dumps({"findings": ws.assumption_checks(), "assumptions": ws.assumptions,
                               "regulatory_parameters": ws.reg_params}), False
        if name == "run_sert":
            s = ws.sert()
            return json.dumps({c: {**v, "scenario_reserves": {k: _k(x) for k, x in v["scenario_reserves"].items()}}
                               for c, v in s.items()} | {"units": "USD thousands", "model_disclaimer": DISCLAIMER}), False
        if name == "run_valuation":
            return json.dumps(_run_summary(ws.valuation(args.get("n_scenarios")))), False
        if name == "run_sensitivity":
            s = ws.sensitivity(args["assumption"], args["shock"], args.get("n_scenarios"))
            s["impact"] = {c: {k: (_k(v) if k.startswith(("base", "shocked", "change")) and k != "change_pct" else v)
                               for k, v in d.items()} for c, d in s["impact"].items()}
            return json.dumps(s | {"units": "USD thousands"}), False
        if name == "get_run":
            run = ws.latest_run(args.get("run_id"))
            if run is None:
                return json.dumps({"error": "no matching run; call run_valuation first"}), True
            return json.dumps(_run_summary(run) | {"assumptions": run["assumptions"],
                                                   "regulatory_parameters": run["regulatory_parameters"]}), False
        if name == "confirm_regulatory_parameter":
            current = ws.reg_params[args["name"]]["value"]
            value = args["value"]
            if isinstance(current, (int, float)) and not isinstance(current, bool):
                value = type(current)(float(value)) if isinstance(current, int) else float(value)
            return json.dumps(ws.confirm_reg_param(args["name"], value, args["citation"])), False
        if name == "save_report":
            ws.save_report(args["kind"], args["title"], args["markdown"])
            return json.dumps({"saved": args["kind"], "title": args["title"],
                               "characters": len(args["markdown"])}), False
    except Exception as exc:  # tool failures go back to the model as error results
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True
    return json.dumps({"error": f"unhandled tool '{name}'"}), True
