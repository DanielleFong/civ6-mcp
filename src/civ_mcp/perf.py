"""Performance instrumentation — APM, blocked time, per-turn yield curves.

Wraps every registered MCP tool with a timing shim and aggregates per-turn
statistics.  Emits two event types through the TelemetryEmitter:

  * EVENT_TOOL_CALL  — one row per tool invocation (name, duration, error)
  * EVENT_TURN_PERF  — one row per completed game turn:
      turn, wall clock, tool calls, actions, APM, time blocked (by category),
      plus the latest empire snapshot (score, yields, cities, pop, districts).

Definitions:
  * "action"  — a write-path tool call (moves, production, research, deals…).
    Read-only queries are overhead, not actions.
  * APM       — actions / wall-clock minutes within the turn window.
  * "blocked" — wall time between an end_turn attempt that failed on a
    blocker (diplomacy session, pending choice…) and the next successful
    end_turn.  Categorised from the error text.

All parsing is regex-over-narration, same approach as evals/metrics.py.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

EVENT_TOOL_CALL = "tool_call"
EVENT_TURN_PERF = "turn_perf"

# Write-path tools count as actions.  Prefix match on tool name.
ACTION_PREFIXES = (
    "move_", "attack", "fortify", "found_", "build_", "promote_", "upgrade_",
    "set_", "purchase", "buy_", "establish", "propose", "respond", "accept",
    "reject", "appoint", "assign", "swap_", "change_", "choose", "recruit",
    "patronize", "activate", "harvest", "repair", "wake_", "delete_",
    "end_turn", "vote", "found", "select_", "launch_", "start_",
)

# Blocker categorisation from end_turn failure text.
BLOCKER_PATTERNS = [
    ("diplomacy", re.compile(r"diplomacy encounter|diplomatic proposal", re.I)),
    ("production", re.compile(r"production|city.*needs", re.I)),
    ("research", re.compile(r"research|tech|civic", re.I)),
    ("unit_orders", re.compile(r"unit.*needs orders|units? await", re.I)),
    ("policy", re.compile(r"policy|government", re.I)),
    ("great_person", re.compile(r"great (person|people)", re.I)),
    ("world_congress", re.compile(r"congress", re.I)),
    ("popup", re.compile(r"popup|dismiss", re.I)),
]

_RE_TURN = re.compile(r"Turn\s+(\d+)")
_RE_SCORE = re.compile(r"Score:\s*(\d+)")
_RE_GOLD = re.compile(r"Gold:\s*(-?[\d.]+)")
_RE_SCI = re.compile(r"Science:\s*([\d.]+)")
_RE_CUL = re.compile(r"Culture:\s*([\d.]+)")
_RE_FAITH = re.compile(r"Faith:\s*([\d.]+)")
_RE_CITIES = re.compile(r"Cities:\s*(\d+)")
_RE_POP = re.compile(r"Population:\s*(\d+)")
_RE_ERA_SCORE = re.compile(r"Era.*?Score:\s*(\d+)")
_RE_DISTRICTS = re.compile(r"Districts:\s*([A-Z_() ,0-9]+)")


def _classify_blocker(text: str) -> str:
    for name, pat in BLOCKER_PATTERNS:
        if pat.search(text):
            return name
    return "other"


class PerfTracker:
    """Aggregates per-tool and per-turn performance stats."""

    def __init__(self, emitter: Any) -> None:
        self._emitter = emitter
        self._turn: int | None = None
        self._turn_started: float = time.monotonic()
        self._calls = 0
        self._errors = 0
        self._actions = 0
        self._tool_time = 0.0
        self._tool_breakdown: dict[str, float] = defaultdict(float)
        self._blocked: dict[str, float] = defaultdict(float)
        self._block_started: float | None = None
        self._block_category: str = "other"
        self._snapshot: dict[str, Any] = {}

    # ── snapshot parsing ─────────────────────────────────────────────

    def _absorb_overview(self, text: str) -> None:
        for key, pat, cast in (
            ("score", _RE_SCORE, int),
            ("gold", _RE_GOLD, float),
            ("science", _RE_SCI, float),
            ("culture", _RE_CUL, float),
            ("faith", _RE_FAITH, float),
            ("cities", _RE_CITIES, int),
            ("population", _RE_POP, int),
            ("era_score", _RE_ERA_SCORE, int),
        ):
            m = pat.search(text)
            if m:
                try:
                    self._snapshot[key] = cast(m.group(1))
                except ValueError:
                    pass

    def _absorb_cities(self, text: str) -> None:
        districts = 0
        for m in _RE_DISTRICTS.finditer(text):
            districts += len([d for d in m.group(1).split() if "(" in d])
        if districts:
            self._snapshot["districts"] = districts

    # ── recording ────────────────────────────────────────────────────

    async def record(
        self, tool: str, duration: float, is_error: bool, result_text: str
    ) -> None:
        self._calls += 1
        self._tool_time += duration
        self._tool_breakdown[tool] += duration
        if is_error:
            self._errors += 1
        if tool.startswith(ACTION_PREFIXES) and not is_error:
            self._actions += 1

        try:
            await self._emitter.emit(
                EVENT_TOOL_CALL,
                {
                    "tool": tool,
                    "duration_s": round(duration, 4),
                    "error": is_error,
                    "turn": self._turn,
                },
            )
        except Exception:
            log.debug("tool_call emit failed", exc_info=True)

        text = result_text or ""
        if tool == "get_game_overview":
            self._absorb_overview(text)
            m = _RE_TURN.search(text)
            if m and self._turn is None:
                self._turn = int(m.group(1))
        elif tool in ("get_cities", "get_city_details"):
            self._absorb_cities(text)
        elif tool == "end_turn":
            await self._handle_end_turn(text, is_error)

    async def _handle_end_turn(self, text: str, is_error: bool) -> None:
        blocked = is_error or "Cannot end turn" in text or "Turn paused" in text
        if blocked:
            if self._block_started is None:
                self._block_started = time.monotonic()
                self._block_category = _classify_blocker(text)
            return

        # Successful end turn — close any open block window and flush the row.
        if self._block_started is not None:
            self._blocked[self._block_category] += (
                time.monotonic() - self._block_started
            )
            self._block_started = None

        now = time.monotonic()
        wall = now - self._turn_started
        minutes = max(wall / 60.0, 1e-9)
        row = {
            "turn": self._turn,
            "wall_s": round(wall, 2),
            "tool_calls": self._calls,
            "tool_errors": self._errors,
            "actions": self._actions,
            "apm": round(self._actions / minutes, 2),
            "calls_per_min": round(self._calls / minutes, 2),
            "tool_time_s": round(self._tool_time, 2),
            "think_time_s": round(max(wall - self._tool_time, 0.0), 2),
            "blocked_s": {k: round(v, 2) for k, v in self._blocked.items()},
            "blocked_total_s": round(sum(self._blocked.values()), 2),
            "slowest_tools": dict(
                sorted(self._tool_breakdown.items(), key=lambda kv: -kv[1])[:5]
            ),
            **{f"snap_{k}": v for k, v in self._snapshot.items()},
        }
        try:
            await self._emitter.emit(EVENT_TURN_PERF, row)
        except Exception:
            log.debug("turn_perf emit failed", exc_info=True)
        log.info(
            "TURN_PERF t%s: %.0fs wall, %d calls, %d actions, %.1f APM, %.0fs blocked",
            self._turn, wall, self._calls, self._actions,
            row["apm"], row["blocked_total_s"],
        )

        # Reset for the next turn window.
        m = _RE_TURN.search(text)
        self._turn = int(m.group(1)) if m else (self._turn + 1 if self._turn else None)
        self._turn_started = now
        self._calls = self._errors = self._actions = 0
        self._tool_time = 0.0
        self._tool_breakdown = defaultdict(float)
        self._blocked = defaultdict(float)


def instrument_tools(mcp: Any, tracker: PerfTracker) -> int:
    """Wrap every registered FastMCP tool with a timing shim.

    Returns the number of tools instrumented.  Idempotent per-process.
    """
    count = 0
    for tool in mcp._tool_manager._tools.values():
        fn = tool.fn
        if getattr(fn, "_perf_wrapped", False):
            continue

        @functools.wraps(fn)
        async def wrapper(*args: Any, _fn=fn, _name=tool.name, **kwargs: Any):
            start = time.monotonic()
            is_error = False
            result: Any = None
            try:
                result = await _fn(*args, **kwargs)
                return result
            except Exception:
                is_error = True
                raise
            finally:
                duration = time.monotonic() - start
                try:
                    await tracker.record(
                        _name, duration, is_error,
                        result if isinstance(result, str) else str(result or ""),
                    )
                except Exception:
                    log.debug("perf record failed for %s", _name, exc_info=True)

        wrapper._perf_wrapped = True  # type: ignore[attr-defined]
        tool.fn = wrapper
        count += 1
    log.info("Perf instrumentation: wrapped %d tools", count)
    return count
