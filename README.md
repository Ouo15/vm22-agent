# VM-22 Valuation Agent

[Live demo](https://vm22-agent.streamlit.app/) · [Blog post](https://ouo15.github.io/posts/vm22-valuation-agent/) · [MIT License](LICENSE)

An agentic AI assistant for **VM-22 principle-based reserving of non-variable annuities**.
Claude orchestrates a tested actuarial engine. It runs the exclusion test and the
deterministic and stochastic reserves, checks data and assumptions, answers regulatory
questions from the NAIC Valuation Manual with page citations, and drafts both a VM-31-style
technical report and a plain-language stakeholder summary.

> **Educational project.** It uses synthetic data, a one-factor interest-rate generator in
> place of the NAIC Generator of Economic Scenarios (GOES), a synthetic mortality table in
> place of prescribed tables, and simplified VM-22 mechanics. It is not a compliant VM-22
> valuation.

## Design principle

**The model never computes reserves.** All numbers come from deterministic, unit-tested
Python code. The agent decides which tools to run, interprets the results, grounds
regulatory statements in the manual's text, and writes the reports.

```
app.py (Streamlit)  ──►  agent/  (Claude Opus 5.5, streaming tool-use loop)
                              │ 10 tools
                              ▼
                         vm22/  (actuarial engine, tests in tests/)
                         data · mortality · scenarios · projection · reserves
                         checks · regulatory search · report export
```

| Tool | What it does |
|---|---|
| `search_valuation_manual` | TF-IDF search over the loaded Valuation Manual, returns passages with chapter and page |
| `describe_portfolio` | Summary of the in-force block |
| `validate_inputs` | Data checks: required fields, ranges, duplicates, outliers |
| `review_assumptions` | Margin, bound and horizon checks; flags unverified regulatory parameters |
| `run_sert` | Stochastic Exclusion Ratio Test per reserving category over 8 rate-stress scenarios |
| `run_valuation` | SERT, Deterministic Reserve, Stochastic Reserve (CTE), floor and reported reserve |
| `run_sensitivity` | One-assumption shocks: mortality, improvement, lapse, crediting spread, expenses, rates |
| `get_run` | Saved run records with the assumptions and parameters used |
| `confirm_regulatory_parameter` | Sets a parameter as verified, but only with a citation from the manual |
| `save_report` | Saves the VM-31 draft or the stakeholder summary (Markdown and Word download) |

## Reserve mechanics (simplified)

- **Products:** single-premium deferred fixed annuities (7-year surrender charge, guaranteed
  minimum rate, dynamic lapses) and payout annuities (life-contingent, optional certain period).
- **Scenario reserve:** the starting assets needed so that a portfolio earning the scenario's
  net earned rate never runs out. Computed as the maximum over time of the cumulative present
  value of net liability cash flows.
- **Deterministic Reserve:** the scenario reserve under the baseline path.
- **Stochastic Reserve:** the CTE (default 70) of scenario reserves over the stochastic paths.
- **Reported reserve:** `max(floor, SR if the SERT fails else DR)`, with a cash-surrender-value
  floor for deferred annuities.

## Regulatory parameters start unverified

The CTE level, SERT threshold, number of scenarios, floor basis and reserve rule live in
[config/regulatory_parameters.yaml](config/regulatory_parameters.yaml) and are all marked
`verified: false`. Upload the Valuation Manual in the app and ask the agent to verify them.
It searches the text, and it can mark a value verified only with a citation. Unverified
parameters are flagged in every result and report.

## Live demo

**[vm22-agent.streamlit.app](https://vm22-agent.streamlit.app/)**, hosted on Streamlit Community Cloud. It runs in **public mode**: you bring your
own Anthropic API key (held in your session's memory only, never stored or logged), and
nothing you upload or change is shared with other visitors. The Results tab works without a key.

## Run it locally

```bash
cd vm22-agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py      # opens http://localhost:8501
```

Paste your API key in the sidebar. To use a key from the environment and save runs, reports
and uploaded PDFs to disk instead, start the app in local mode:

```bash
export ANTHROPIC_API_KEY=...
VM22_LOCAL_MODE=1 .venv/bin/streamlit run app.py
```

1. Download the current **NAIC Valuation Manual** PDF from naic.org and upload it in the
   sidebar (Valuation Manual → Index PDF). The PDF is git-ignored.
2. In the **Agent** tab, try for example: *"Run a full VM-22 valuation and explain the
   results"*, *"Verify the CTE level and SERT threshold against the manual"*, *"Draft the
   VM-31 report and the stakeholder summary"*.
3. The **Results** tab works without an API key: run checks, the SERT, valuations and
   sensitivities, and see charts.
4. The **Reports** tab shows the drafts with Markdown and Word downloads.

Run the tests with `.venv/bin/python -m pytest`.

### Public mode vs local mode

| | Public mode (default) | Local mode (`VM22_LOCAL_MODE=1`) |
|---|---|---|
| API key | Visitor's own key only; server keys are ignored | Visitor's key, or `ANTHROPIC_API_KEY` |
| Uploaded PDFs | Private temporary folder per session | `data/regulations/` |
| Verified parameters | This session only | Saved to `config/regulatory_parameters.yaml` |
| Runs and reports | In memory | Also saved to `outputs/` |

## Agent implementation notes

- Claude API manual streaming loop with `claude-opus-5-5`, adaptive thinking at `high`
  effort, and progress notes between tool calls (`thinking.display: "updates"`).
- `fallbacks: "default"`: if a request is declined, the API retries it on a fallback model;
  refusals are handled explicitly.
- Tools stream their inputs (`eager_input_streaming`) and every input is validated against
  its schema before the tool runs.
- The system prompt and tool list are stable so prompt caching works across turns.

## Possible extensions

- Plug in the prescribed GOES scenario files and prescribed mortality tables.
- Add reserving categories and products (fixed indexed annuities, longevity reinsurance).
- Asset modeling with reinvestment guardrails, and the NAER-based discounting.
- An evaluation set of regulatory questions with known answers to measure citation accuracy.

## License

[MIT](LICENSE). You're free to use, modify and share the code, with attribution.
