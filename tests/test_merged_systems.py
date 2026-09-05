"""Interaction tests for the systems merged on 2026-09-05 (LIG-982/987 + connector bootstrap).

No game needed: Connection scanning, execute_plan batching, kill_game ownership
policy, tuner-port override, and the cinematic auto-dismiss list are exercised
with stubs. Each test states the behaviour it pins.
"""

import asyncio
import importlib
import json
import sys
import types

import pytest

from civ_mcp import connection as conn_mod
from civ_mcp import game_launcher as gl


# ---------------------------------------------------------------------------
# Connection: CIV_MCP_EXPECTED_GAME scanning (LIG-987) x bootstrap claim
# ---------------------------------------------------------------------------

def _conn(port_identity: dict, monkeypatch):
    """Connection whose _connect_once succeeds on ports in port_identity and whose
    probe returns the mapped value ('' = menu, None = probe failed, str = game)."""
    c = conn_mod.GameConnection.__new__(conn_mod.GameConnection)
    c.host, c.port = "127.0.0.1", 4318
    c.ingame_index = None
    c.log = []

    async def _connect_once():
        if c.port not in port_identity:
            raise ConnectionError("no tuner")
        c.log.append(("connect", c.port))
        c.ingame_index = None if port_identity[c.port] == "" else 94

    async def _identity_probe():
        return port_identity[c.port]

    async def disconnect():
        c.log.append(("disconnect", c.port))

    c._connect_once, c._identity_probe, c.disconnect = _connect_once, _identity_probe, disconnect
    return c


class TestExpectedGameScan:
    def test_skips_wrong_game_and_binds_to_the_right_port(self, monkeypatch):
        c = _conn({4318: "civilization_england|leader_victoria", 4319: "civilization_russia|leader_peter"}, monkeypatch)
        asyncio.run(c._connect_to_expected_game("russia"))
        assert c.port == 4319
        assert ("disconnect", 4318) in c.log  # wrong game released

    def test_menu_stage_instance_is_claimable_for_bootstrap(self, monkeypatch):
        c = _conn({4318: ""}, monkeypatch)  # tuner up, no game loaded
        asyncio.run(c._connect_to_expected_game("russia"))
        assert c.port == 4318

    def test_probe_failure_is_never_a_match(self, monkeypatch):
        """A loaded game whose probe timed out must NOT be claimed — that is how the
        MCP attaches to a human's game during a slow AI turn."""
        c = _conn({4318: None}, monkeypatch)
        with pytest.raises(ConnectionError) as ei:
            asyncio.run(c._connect_to_expected_game("russia"))
        assert "probe failed" in str(ei.value)
        assert ("disconnect", 4318) in c.log

    def test_nothing_listening_reports_scan(self, monkeypatch):
        c = _conn({}, monkeypatch)
        with pytest.raises(ConnectionError) as ei:
            asyncio.run(c._connect_to_expected_game("russia"))
        assert "not found" in str(ei.value)
        assert c.port == 4318  # restored to base


# ---------------------------------------------------------------------------
# tuner_client: CIV_MCP_TUNER_PORT overrides the default (read at import)
# ---------------------------------------------------------------------------

def test_tuner_port_env_override(monkeypatch):
    import civ_mcp.tuner_client as tc
    monkeypatch.setenv("CIV_MCP_TUNER_PORT", "4402")
    importlib.reload(tc)
    try:
        assert tc.DEFAULT_PORT == 4402
        c = conn_mod.GameConnection()
        assert c.port == 4402
    finally:
        monkeypatch.delenv("CIV_MCP_TUNER_PORT")
        importlib.reload(tc)
        assert tc.DEFAULT_PORT == 4318


# ---------------------------------------------------------------------------
# execute_plan (LIG-982): batching, ordering rules, stop-on-failure
# ---------------------------------------------------------------------------

@pytest.fixture
def plan_env(monkeypatch):
    """Stub the four ops behind execute_plan; record calls."""
    import civ_mcp.server as srv
    calls = []

    def mk(name, result="OK"):
        async def op(ctx, **kw):
            calls.append((name, kw))
            return result() if callable(result) else result
        return op

    monkeypatch.setattr(srv, "unit_action", mk("unit_action"))
    monkeypatch.setattr(srv, "set_city_production", mk("set_city_production", "Error: no such city"))
    monkeypatch.setattr(srv, "set_research", mk("set_research"))
    monkeypatch.setattr(srv, "end_turn", mk("end_turn", "Turn 12 -> 13\nreport"))
    return srv, calls


def _run(srv, plan):
    return asyncio.run(srv.execute_plan(None, json.dumps(plan) if not isinstance(plan, str) else plan))


class TestExecutePlan:
    def test_parse_errors(self, plan_env):
        srv, _ = plan_env
        assert _run(srv, "not json").startswith("PLAN PARSE ERROR")
        assert _run(srv, []).startswith("PLAN PARSE ERROR")
        assert _run(srv, [{"op": "set_research", "tech_or_civic": "TECH_MINING"}] * 31).startswith("PLAN PARSE ERROR")

    def test_serial_execution_and_end_turn_report(self, plan_env):
        srv, calls = plan_env
        out = _run(srv, [
            {"op": "unit_action", "unit_id": 1, "action": "move", "target_x": 5, "target_y": 7},
            {"op": "set_research", "tech_or_civic": "TECH_MINING"},
            {"op": "end_turn", "tactical": "t"},
        ])
        assert [c[0] for c in calls] == ["unit_action", "set_research", "end_turn"]
        assert calls[0][1] == {"unit_id": 1, "action": "move", "target_x": 5, "target_y": 7}  # op stripped
        assert "[2] end_turn:" in out and "Turn 12 -> 13" in out

    def test_end_turn_must_be_last(self, plan_env):
        srv, calls = plan_env
        out = _run(srv, [{"op": "end_turn"}, {"op": "set_research", "tech_or_civic": "X"}])
        assert "REJECTED" in out and "[1] set_research: SKIPPED" in out
        assert calls == []  # nothing executed

    def test_failure_stops_and_marks_rest_skipped(self, plan_env):
        srv, calls = plan_env
        out = _run(srv, [
            {"op": "set_city_production", "city_id": 1, "item_type": "UNIT", "item_name": "UNIT_SETTLER"},
            {"op": "set_research", "tech_or_civic": "TECH_MINING"},
            {"op": "end_turn"},
        ])
        assert "[0] set_city_production: Error: no such city" in out
        assert "[1] set_research: SKIPPED" in out and "[2] end_turn: SKIPPED" in out
        assert [c[0] for c in calls] == ["set_city_production"]  # end_turn never sent

    def test_unsupported_op_stops(self, plan_env):
        srv, calls = plan_env
        out = _run(srv, [{"op": "found_city"}, {"op": "end_turn"}])
        assert "UNSUPPORTED" in out and "[1] end_turn: SKIPPED" in out and calls == []

    def test_exception_in_op_is_reported_not_raised(self, plan_env, monkeypatch):
        srv, calls = plan_env

        async def boom(ctx, **kw):
            raise RuntimeError("tuner gone")
        monkeypatch.setattr(srv, "unit_action", boom)
        out = _run(srv, [{"op": "unit_action", "unit_id": 1}, {"op": "end_turn"}])
        assert "ERROR: tuner gone" in out and "[1] end_turn: SKIPPED" in out


# ---------------------------------------------------------------------------
# kill_game (win32): never kill a human's instance
# ---------------------------------------------------------------------------

@pytest.fixture
def win_kill(monkeypatch):
    runs = []
    monkeypatch.setattr(gl.sys, "platform", "win32")
    monkeypatch.setattr(gl, "is_game_running", lambda: True)
    monkeypatch.setattr(gl.time, "sleep", lambda s: None)
    monkeypatch.setattr(gl.subprocess, "run", lambda args, **kw: runs.append(args) or types.SimpleNamespace(stdout="", returncode=0))
    monkeypatch.setattr(gl, "_PROCESS_NAMES", ("CivilizationVI_DX12.exe",))
    monkeypatch.delenv("CIV_MCP_KILL_ALL", raising=False)
    gl._LAUNCHED_PID[:] = []
    return runs


class TestKillGameOwnership:
    def test_two_instances_unknown_owner_refuses(self, win_kill, monkeypatch):
        monkeypatch.setattr(gl, "_win32_civ6_pids", lambda: [111, 222])
        out = gl._kill_game_sync()
        assert out.startswith("REFUSED") and win_kill == []

    def test_two_instances_known_owner_kills_only_ours(self, win_kill, monkeypatch):
        monkeypatch.setattr(gl, "_win32_civ6_pids", lambda: [111, 222])
        gl._LAUNCHED_PID[:] = [222]
        gl._kill_game_sync()
        assert win_kill == [["taskkill", "/PID", "222", "/F"]]

    def test_owner_pid_gone_refuses(self, win_kill, monkeypatch):
        """Our recorded PID died; two others remain — still unknown, still refuse."""
        monkeypatch.setattr(gl, "_win32_civ6_pids", lambda: [111, 222])
        gl._LAUNCHED_PID[:] = [999]
        assert gl._kill_game_sync().startswith("REFUSED") and win_kill == []

    def test_single_instance_kills_by_image(self, win_kill, monkeypatch):
        monkeypatch.setattr(gl, "_win32_civ6_pids", lambda: [111])
        gl._kill_game_sync()
        assert win_kill == [["taskkill", "/IM", "CivilizationVI_DX12.exe", "/F"]]

    def test_kill_all_env_overrides(self, win_kill, monkeypatch):
        monkeypatch.setattr(gl, "_win32_civ6_pids", lambda: [111, 222])
        monkeypatch.setenv("CIV_MCP_KILL_ALL", "1")
        gl._kill_game_sync()
        assert win_kill and win_kill[0][1] == "/IM"


# ---------------------------------------------------------------------------
# spectator: choice-free cinematics are on the auto-dismiss list
# ---------------------------------------------------------------------------

def test_cinematics_auto_dismissed_but_choices_not():
    from civ_mcp import spectator
    lst = spectator._NONCRITICAL_POPUPS
    assert "HeroesPopup" in lst and "SecretSocietyPopup" in lst
    # Screens that carry a decision must never be on this list.
    for decision_screen in ("WorldCongressPopup", "DiplomacyActionView", "DiplomacyDealView", "EspionageEscape", "GovernorPromotionPopup"):
        assert decision_screen not in lst
