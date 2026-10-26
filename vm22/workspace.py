"""Session state shared by the app and the agent's tools."""

import copy
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import checks, data, reserves
from .regulatory import RegulatoryIndex

ROOT = Path(__file__).resolve().parent.parent


class Workspace:
    """persist=False keeps everything in this session: uploaded PDFs go to a private
    temporary folder, and parameter changes, runs and reports are never written to the
    shared project folder. Use it whenever several people share one server."""

    def __init__(self, root=ROOT, persist=True):
        self.root = Path(root)
        self.persist = persist
        self.assumptions = yaml.safe_load((self.root / "config/assumptions.yaml").read_text())
        self.reg_params = yaml.safe_load((self.root / "config/regulatory_parameters.yaml").read_text())
        self.portfolio = {"deferred": data.synthetic_deferred(), "payout": data.synthetic_payout()}
        self.portfolio_source = "synthetic"
        self.runs = []          # valuation run records, newest last
        self.sert_result = None
        self.sensitivities = []
        self.reports = {}       # kind -> {"title", "markdown", "saved_at"}
        self.reg_dir = self.root / "data/regulations" if persist else Path(tempfile.mkdtemp(prefix="vm22-regs-"))
        self.out_dir = self.root / "outputs"
        self.reg_index = RegulatoryIndex([])
        self.reload_regulations()

    # --- regulations -------------------------------------------------------
    def reload_regulations(self):
        self.reg_dir.mkdir(parents=True, exist_ok=True)
        self.reg_index = RegulatoryIndex.from_folder(self.reg_dir)

    # --- parameters --------------------------------------------------------
    def save_reg_params(self):
        if not self.persist:
            return
        (self.root / "config/regulatory_parameters.yaml").write_text(yaml.safe_dump(self.reg_params, sort_keys=False))

    def confirm_reg_param(self, name, value, citation):
        if name not in self.reg_params:
            raise ValueError(f"unknown parameter '{name}'")
        old = self.reg_params[name]["value"]
        self.reg_params[name].update(value=value, verified=True, source=citation)
        self.save_reg_params()
        return {"name": name, "old_value": old, "new_value": value, "source": citation}

    # --- engine wrappers -----------------------------------------------------
    def input_checks(self):
        return checks.check_inputs(self.portfolio)

    def assumption_checks(self):
        return checks.check_assumptions(self.assumptions, self.reg_params, self.portfolio)

    def sert(self):
        self.sert_result = reserves.run_sert(self.portfolio, self.assumptions, self.reg_params)
        return self.sert_result

    def valuation(self, n_scenarios=None):
        run = reserves.run_valuation(self.portfolio, self.assumptions, self.reg_params, n_scenarios)
        run["assumptions"] = copy.deepcopy(self.assumptions)
        run["regulatory_parameters"] = copy.deepcopy(self.reg_params)
        run["portfolio_source"] = self.portfolio_source
        run["checks"] = checks.summarize(self.input_checks() + self.assumption_checks())
        self.runs.append(run)
        self.sert_result = run["sert"]
        self._write(f"run_{run['run_id']}.json", json.dumps(run, indent=1))
        return run

    def sensitivity(self, assumption, shock, n_scenarios=None):
        res = reserves.run_sensitivity(self.portfolio, self.assumptions, self.reg_params, assumption, shock, n_scenarios)
        self.sensitivities.append(res)
        return res

    def save_report(self, kind, title, markdown):
        self.reports[kind] = {"title": title, "markdown": markdown,
                              "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self._write(f"{kind}.md", f"# {title}\n\n{markdown}")
        return self.reports[kind]

    def latest_run(self, run_id=None):
        if not self.runs:
            return None
        if run_id is None:
            return self.runs[-1]
        return next((r for r in self.runs if r["run_id"] == run_id), None)

    def _write(self, name, text):
        if not self.persist:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / name).write_text(text)
