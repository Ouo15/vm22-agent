"""Streamlit front end for the VM-22 valuation agent.

Run with:  streamlit run app.py
"""

import os

import anthropic
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent.agent import MODEL, VM22Agent
from vm22 import DISCLAIMER, data
from vm22.reports import REPORT_KINDS, markdown_to_docx
from vm22.reserves import SENSITIVITY_UNITS
from vm22.workspace import Workspace

BLUE, GREY, RED = "#06436e", "#9ca3af", "#b2182b"
st.set_page_config(page_title="VM-22 Valuation Agent", page_icon="📊", layout="wide")

if "ws" not in st.session_state:
    st.session_state.ws = Workspace()
    st.session_state.chat = []      # display history: {"role", "text", "steps"}
    st.session_state.agent = None
ws = st.session_state.ws


def money(x):
    return f"${x / 1e6:,.1f}M"


def get_agent():
    if st.session_state.agent is None:
        key = st.session_state.get("api_key") or None
        if not (key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return None
        st.session_state.agent = VM22Agent(ws, anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic())
    return st.session_state.agent


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("VM-22 Valuation Agent")
    st.caption(DISCLAIMER)

    with st.expander("Claude API", expanded=not os.environ.get("ANTHROPIC_API_KEY")):
        if os.environ.get("ANTHROPIC_API_KEY"):
            st.success(f"Using ANTHROPIC_API_KEY · model {MODEL}")
        else:
            st.text_input("API key (kept in this session only)", type="password", key="api_key",
                          on_change=lambda: st.session_state.update(agent=None))

    with st.expander("In-force data", expanded=False):
        st.write(f"Source: **{ws.portfolio_source}**")
        if st.button("Generate new synthetic block"):
            seed = int(np.random.default_rng().integers(1, 10_000))
            ws.portfolio = {"deferred": data.synthetic_deferred(seed=seed), "payout": data.synthetic_payout(seed=seed + 1)}
            ws.portfolio_source = f"synthetic (seed {seed})"
            st.rerun()
        up_d = st.file_uploader("Deferred annuities CSV", type="csv", help=", ".join(data.DEFERRED_COLUMNS))
        up_p = st.file_uploader("Payout annuities CSV", type="csv", help=", ".join(data.PAYOUT_COLUMNS))
        if (up_d or up_p) and st.button("Load uploaded data"):
            ws.portfolio = {"deferred": pd.read_csv(up_d) if up_d else None, "payout": pd.read_csv(up_p) if up_p else None}
            ws.portfolio_source = "uploaded CSV"
            st.rerun()

    with st.expander("Valuation Manual", expanded=not ws.reg_index.records):
        st.caption("Download the current Valuation Manual from naic.org and upload the PDF here. "
                   "The agent cites it when answering regulatory questions.")
        pdf = st.file_uploader("Regulatory PDF", type="pdf")
        if pdf and st.button("Index PDF"):
            (ws.reg_dir / pdf.name).write_bytes(pdf.getvalue())
            with st.spinner("Indexing..."):
                ws.reload_regulations()
            st.rerun()
        for doc, chapters in ws.reg_index.documents.items():
            main = {k: v for k, v in chapters.items() if k in ("VM-20", "VM-21", "VM-22", "VM-31")}
            st.write(f"**{doc}**: " + (", ".join(f"{k} ({v} passages)" for k, v in sorted(main.items())) or "indexed"))

    with st.expander("Regulatory parameters", expanded=False):
        st.dataframe(pd.DataFrame([{"parameter": k, "value": str(v["value"]),
                                    "verified": "✅" if v.get("verified") else "⚠️"}
                                   for k, v in ws.reg_params.items()]), hide_index=True, width="stretch")
        st.caption("⚠️ = not yet confirmed against the Valuation Manual.")

    with st.expander("Key assumptions", expanded=False):
        a = ws.assumptions
        a["company"]["crediting_spread"] = st.number_input("Crediting spread", 0.0, 0.05, a["company"]["crediting_spread"], 0.0025, format="%.4f")
        a["policyholder_behavior"]["lapse_margin"] = st.number_input("Lapse margin", -0.5, 0.5, a["policyholder_behavior"]["lapse_margin"], 0.05)
        a["mortality"]["margin_payout"] = st.number_input("Payout mortality margin", 0.0, 0.3, a["mortality"]["margin_payout"], 0.01)
        a["mortality"]["improvement_rate"] = st.number_input("Mortality improvement", 0.0, 0.03, a["mortality"]["improvement_rate"], 0.001, format="%.3f")
        a["economic"]["initial_short_rate"] = st.number_input("Initial short rate", -0.01, 0.10, a["economic"]["initial_short_rate"], 0.0025, format="%.4f")
        a["economic"]["volatility"] = st.number_input("Rate volatility", 0.001, 0.03, a["economic"]["volatility"], 0.001, format="%.3f")

tab_chat, tab_results, tab_reports = st.tabs(["💬 Agent", "📈 Results", "📄 Reports"])

# --------------------------------------------------------------------------- chat
with tab_chat:
    agent = get_agent()
    if agent is None:
        st.info("Add a Claude API key in the sidebar (or set ANTHROPIC_API_KEY) to use the agent. "
                "The Results tab works without one.")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            if msg.get("steps"):
                with st.expander(f"{len(msg['steps'])} steps", expanded=False):
                    for s in msg["steps"]:
                        st.caption(s)
            st.markdown(msg["text"])

    examples = ["Run a full VM-22 valuation and explain the results",
                "Which reserving categories need a stochastic reserve, and why?",
                "What does VM-22 require for the Stochastic Exclusion Ratio Test?",
                "Draft the VM-31 report and the stakeholder summary"]
    cols = st.columns(len(examples))
    clicked = next((ex for c, ex in zip(cols, examples) if c.button(ex, disabled=agent is None, width="stretch")), None)
    prompt = st.chat_input("Ask the VM-22 agent...", disabled=agent is None) or clicked

    if prompt and agent is not None:
        st.session_state.chat.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            status = st.status("Working...", expanded=True)
            placeholder = st.empty()
            text, steps, after_tool = "", [], False
            for ev in agent.run_turn(prompt):
                if ev["type"] == "text":
                    if after_tool and text:
                        text += "\n\n"
                    after_tool = False
                    text += ev["text"]
                    placeholder.markdown(text + "▌")
                elif ev["type"] == "status":
                    steps.append(f"💭 {ev['text']}")
                    status.write(f"💭 {ev['text']}")
                elif ev["type"] == "tool_start":
                    status.update(label=f"Running {ev['name']}...")
                elif ev["type"] == "tool_end":
                    mark = "❌" if ev["is_error"] else "🔧"
                    steps.append(f"{mark} {ev['name']}")
                    status.write(f"{mark} `{ev['name']}`")
                    after_tool = True
                elif ev["type"] in ("notice", "error"):
                    text += f"\n\n> {ev['text']}"
            status.update(label=f"Done · {len(steps)} steps", state="complete", expanded=False)
            placeholder.markdown(text)
        st.session_state.chat.append({"role": "assistant", "text": text, "steps": steps})

# --------------------------------------------------------------------------- results
with tab_results:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    n = c4.select_slider("Stochastic scenarios", [100, 250, 500, 1000, 2000],
                         value=int(ws.reg_params["n_stochastic_scenarios"]["value"]))
    if c1.button("Validate inputs", width="stretch"):
        st.session_state.findings = ws.input_checks() + ws.assumption_checks()
    if c2.button("Run SERT", width="stretch"):
        ws.sert()
    if c3.button("Run valuation", type="primary", width="stretch"):
        with st.spinner("Projecting cash flows..."):
            ws.valuation(n)

    run = ws.latest_run()
    if run:
        tot = run["results"]["total"]
        st.subheader(f"Run {run['run_id']} · {run['timestamp']}")
        m = st.columns(4)
        m[0].metric("Total reported reserve", money(tot["reported_reserve"]))
        for i, cat in enumerate(("deferred", "payout"), start=1):
            if cat in run["results"]:
                r = run["results"][cat]
                m[i].metric(f"{cat.title()} reported", money(r["reported_reserve"]),
                            "floor binds" if r["floor_binding"] else ("stochastic" if r["stochastic_required"] else "deterministic"),
                            delta_color="off")
        m[3].metric("Scenarios / CTE", f"{run['n_scenarios']} / {run['cte_level']:.0%}")
        if run["unverified_parameters"]:
            st.warning("Unverified regulatory parameters: " + ", ".join(run["unverified_parameters"]))

        rows = []
        for cat in ("deferred", "payout"):
            if cat in run["results"]:
                r = run["results"][cat]
                rows.append({"category": cat, "DR": money(r["deterministic_reserve"]), "SR (CTE)": money(r["stochastic_reserve"]),
                             "SERT ratio": f"{run['sert'][cat]['ratio']:.2%}", "stochastic required": r["stochastic_required"],
                             "floor": money(r["floor"]), "reported": money(r["reported_reserve"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        g1, g2 = st.columns(2)
        with g1:
            cat = st.radio("Scenario reserve distribution", [c for c in ("deferred", "payout") if c in run["results"]], horizontal=True)
            r = run["results"][cat]
            fig = go.Figure(go.Histogram(x=np.array(r["scenario_reserves"]) / 1e6, nbinsx=40, marker_color=BLUE, opacity=0.75))
            for x, label, color in ((r["deterministic_reserve"], "DR", GREY), (r["stochastic_reserve"], "SR", RED), (r["floor"], "Floor", "black")):
                if x > 0:
                    fig.add_vline(x=x / 1e6, line_color=color, line_dash="dash", annotation_text=label)
            fig.update_layout(xaxis_title="Scenario reserve ($M)", yaxis_title="Scenarios", height=360, margin=dict(t=30, b=10))
            st.plotly_chart(fig, width="stretch")
        with g2:
            fig = go.Figure()
            for cat, color in (("deferred", BLUE), ("payout", GREY)):
                if cat in run["results"]:
                    cf = np.array(run["results"][cat]["baseline_cash_flows"]) / 1e6
                    fig.add_trace(go.Bar(x=np.arange(1, len(cf) + 1), y=cf, name=cat, marker_color=color))
            fig.update_layout(barmode="stack", title="Baseline liability cash flows", xaxis_title="Projection year",
                              yaxis_title="$M", height=400, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch")

    if ws.sert_result:
        with st.expander("SERT stress scenario reserves"):
            st.dataframe(pd.DataFrame({c: {k: money(v) for k, v in s["scenario_reserves"].items()}
                                       for c, s in ws.sert_result.items()}), width="stretch")

    with st.expander("Sensitivity analysis"):
        s1, s2, s3 = st.columns([2, 1, 1])
        which = s1.selectbox("Assumption", list(SENSITIVITY_UNITS), help="; ".join(f"{k}: {v}" for k, v in SENSITIVITY_UNITS.items()))
        shock = s2.number_input("Shock", -0.5, 0.5, 0.10, 0.01)
        if s3.button("Run sensitivity"):
            with st.spinner("Running..."):
                ws.sensitivity(which, shock, n)
        if ws.sensitivities:
            st.dataframe(pd.DataFrame([{"assumption": s["assumption"], "shock": s["shock"],
                                        **{f"{c} change": money(d["change"]) for c, d in s["impact"].items()}}
                                       for s in ws.sensitivities]), hide_index=True, width="stretch")

    if st.session_state.get("findings"):
        with st.expander("Input and assumption checks", expanded=True):
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
            for f in st.session_state.findings:
                st.write(f"{icon[f['severity']]} **{f['check']}**: {f['detail']}")

# --------------------------------------------------------------------------- reports
with tab_reports:
    if not ws.reports:
        st.info("No reports yet. Ask the agent to draft the VM-31 report or the stakeholder summary.")
    for kind, rep in ws.reports.items():
        st.subheader(rep["title"])
        st.caption(f"{REPORT_KINDS[kind]} · saved {rep['saved_at']}")
        d1, d2, _ = st.columns([1, 1, 4])
        d1.download_button("Markdown", f"# {rep['title']}\n\n{rep['markdown']}", file_name=f"{kind}.md", key=f"md-{kind}")
        d2.download_button("Word (.docx)", markdown_to_docx(rep["markdown"], rep["title"]), file_name=f"{kind}.docx", key=f"docx-{kind}")
        with st.container(border=True):
            st.markdown(rep["markdown"])
