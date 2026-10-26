import json
from types import SimpleNamespace as NS

import pytest

from agent import tools
from agent.agent import VM22Agent, _blocks_to_send
from vm22.workspace import Workspace


@pytest.fixture
def ws():
    return Workspace(persist=False)  # nothing is written to the project folder


def test_validate_rejects_bad_input():
    assert tools.validate("run_sensitivity", {"assumption": "lapse"}) == "missing required field 'shock'"
    assert "one of" in tools.validate("run_sensitivity", {"assumption": "nope", "shock": 0.1})
    assert "out of range" in tools.validate("run_sensitivity", {"assumption": "lapse", "shock": 2})
    assert "unexpected" in tools.validate("describe_portfolio", {"x": 1})
    assert "must be integer" in tools.validate("run_valuation", {"n_scenarios": True})
    assert tools.validate("run_valuation", {"n_scenarios": 100}) is None


def test_execute_tools(ws):
    text, err = tools.execute(ws, "run_valuation", {"n_scenarios": 100})
    out = json.loads(text)
    assert not err and out["units"] == "USD thousands" and "deferred" in out["results"]
    text, err = tools.execute(ws, "search_valuation_manual", {"query": "SERT"})
    assert err and "No Valuation Manual" in json.loads(text)["error"]
    text, err = tools.execute(ws, "confirm_regulatory_parameter",
                              {"name": "n_stochastic_scenarios", "value": 1000.0, "citation": "VM-22 p.1"})
    assert not err and ws.reg_params["n_stochastic_scenarios"] == {**ws.reg_params["n_stochastic_scenarios"],
                                                                     "value": 1000, "verified": True}
    text, err = tools.execute(ws, "save_report", {"kind": "stakeholder_summary", "title": "T", "markdown": "## A\n- b"})
    assert not err and ws.reports["stakeholder_summary"]["title"] == "T"


def test_blocks_after_fallback_drop_internal_blocks():
    content = [NS(type="thinking"), NS(type="tool_use"), NS(type="text"), NS(type="fallback"),
               NS(type="thinking"), NS(type="tool_use")]
    assert [b.type for b in _blocks_to_send(content)] == ["text", "fallback", "thinking", "tool_use"]


class FakeStream:
    def __init__(self, message, events):
        self.message, self.events = message, events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.message


class FakeClient:
    """Scripted responses: first a tool call, then a final text answer."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return self.script.pop(0)


def test_agent_loop_runs_tool_then_answers(ws):
    tool_block = NS(type="tool_use", id="tu_1", name="describe_portfolio", input={})
    first = FakeStream(NS(content=[tool_block], stop_reason="tool_use", model="m"),
                       [NS(type="content_block_start", content_block=tool_block)])
    final_text = NS(type="text", text="The block has 1,000 deferred contracts.")
    second = FakeStream(NS(content=[final_text], stop_reason="end_turn", model="m"),
                        [NS(type="text", text=final_text.text)])
    client = FakeClient([first, second])
    agent = VM22Agent(ws, client)

    events = list(agent.run_turn("Describe the portfolio"))
    kinds = [e["type"] for e in events]
    assert kinds == ["tool_start", "tool_end", "text", "done"]
    # tool result was sent back on the second call, matched to the tool_use id
    history = client.calls[1]["messages"]   # same list object; the final answer was appended after the call
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    sent = history[2]["content"][0]
    assert sent["tool_use_id"] == "tu_1" and not sent["is_error"]
    assert json.loads(sent["content"])["deferred"]["contracts"] == 1000
    assert client.calls[0]["model"] == "claude-opus-5-5" and client.calls[0]["fallbacks"] == "default"


def test_agent_stops_on_refusal(ws):
    refusal = FakeStream(NS(content=[], stop_reason="refusal", model="m"), [])
    agent = VM22Agent(ws, FakeClient([refusal]))
    assert [e["type"] for e in agent.run_turn("hi")] == ["error"]
