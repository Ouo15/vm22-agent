"""Public (hosted) mode must never use a server-side key or write to the shared project folder."""

from vm22.workspace import ROOT, Workspace


def _snapshot():
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts
             and ".pytest_cache" not in p.parts and ".git" not in p.parts]
    return {str(p): p.stat().st_mtime_ns for p in files}


def test_public_workspace_writes_nothing():
    before = _snapshot()
    ws = Workspace(persist=False)
    ws.confirm_reg_param("sr_cte_level", 0.7, "test citation")
    ws.valuation(n_scenarios=50)
    ws.save_report("vm31_draft", "T", "body")
    assert ws.reg_params["sr_cte_level"]["verified"]           # changed in this session only
    assert not Workspace(persist=False).reg_params["sr_cte_level"]["verified"]
    assert ROOT not in ws.reg_dir.parents                      # uploads go to a private temp folder
    assert _snapshot() == before


def _sidebar_text(at):
    return " ".join(str(getattr(e, "value", "")) for e in list(at.sidebar.success) + list(at.sidebar.caption))


def test_app_public_mode_ignores_server_key(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-server-key-must-not-be-used")
    monkeypatch.delenv("VM22_LOCAL_MODE", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.session_state.agent is None and at.chat_input[0].disabled
    assert "Local mode" not in _sidebar_text(at) and "Bring your own key" in _sidebar_text(at)


def test_app_local_mode_uses_env_key(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-local")
    monkeypatch.setenv("VM22_LOCAL_MODE", "1")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.session_state.agent is not None and not at.chat_input[0].disabled
    assert "Local mode" in _sidebar_text(at)
