#!/usr/bin/env python3
"""CivBench Orchestrator — dispatch, monitor, and manage benchmark runs.

Manages a fleet of machines running CivBench scenarios via SSH.
Config-driven (no hardcoded IPs/paths), with health monitoring,
failure recovery, and automatic result collection.

Usage:
    python scripts/orchestrator.py preflight [--machines M1,M2]
    python scripts/orchestrator.py launch --scenarios S --models M --runs N --machines M1,M2
    python scripts/orchestrator.py status
    python scripts/orchestrator.py summary
    python scripts/orchestrator.py sync [--machines M1,M2]
    python scripts/orchestrator.py logs --machine M [--last N] [--errors]
    python scripts/orchestrator.py kill-all
    python scripts/orchestrator.py resume

Config: ~/.civbench/machines.yaml
State:  ~/.civbench/state.json
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("orchestrator")

CONFIG_DIR = Path.home() / ".civbench"
CONFIG_PATH = CONFIG_DIR / "machines.yaml"
STATE_PATH = CONFIG_DIR / "state.json"

# ---------------------------------------------------------------------------
# ETA prediction — piecewise turn-time model with Bayesian blending
# ---------------------------------------------------------------------------

# Segments: (start_turn, end_turn) matching Civ VI game phases
_ETA_SEGMENTS = [(1, 50), (50, 150), (150, 330)]

# Per-model priors: (early, mid, late) min/turn, expected end turn, stall overhead
_ETA_PRIORS: dict[str, dict[str, Any]] = {
    "opus": {"rates": (1.5, 2.5, 3.5), "end_turn": 300, "stall_mult": 1.08},
    "gpt": {"rates": (1.8, 2.4, 3.3), "end_turn": 290, "stall_mult": 1.10},
    "gemini": {"rates": (3.0, 5.0, 7.0), "end_turn": 280, "stall_mult": 1.12},
    "sonnet": {"rates": (1.3, 2.2, 3.0), "end_turn": 300, "stall_mult": 1.08},
}
_ETA_DEFAULT = {"rates": (2.0, 3.5, 5.0), "end_turn": 290, "stall_mult": 1.10}
_ETA_PSEUDOCOUNT = 20  # prior strength — ~20 observed turns to halve prior influence


def _eta_integrate(rates: tuple, from_turn: int, to_turn: int) -> float:
    """Total minutes from from_turn to to_turn given piecewise segment rates."""
    total = 0.0
    for (seg_s, seg_e), rate in zip(_ETA_SEGMENTS, rates):
        lo = max(from_turn, seg_s)
        hi = min(to_turn, seg_e)
        if hi > lo:
            total += (hi - lo) * rate
    return total


def estimate_eta(
    model_name: str, current_turn: int, elapsed_h: float, turn_limit: int = 330
) -> dict[str, float]:
    """Non-linear ETA with confidence interval.

    Returns ``{"eta_h": 11.9, "lo_h": 7.2, "hi_h": 18.1}``.
    """
    # Match model to prior
    model_lower = model_name.lower()
    prior = _ETA_DEFAULT
    for key, p in _ETA_PRIORS.items():
        if key in model_lower:
            prior = p
            break

    pr = prior["rates"]
    elapsed_min = elapsed_h * 60

    # Scale factor: how does observed speed compare to prior expectation?
    prior_elapsed = _eta_integrate(pr, 1, current_turn)
    scale = elapsed_min / prior_elapsed if prior_elapsed > 0 else 1.0

    # Blend observed data with prior per segment
    rates = []
    for i, ((seg_s, seg_e), r) in enumerate(zip(_ETA_SEGMENTS, pr)):
        n_obs = max(0, min(current_turn, seg_e) - seg_s)
        obs_rate = r * scale if n_obs > 0 else r
        blended = (n_obs * obs_rate + _ETA_PSEUDOCOUNT * r) / (n_obs + _ETA_PSEUDOCOUNT)
        rates.append(blended)

    # Point estimate
    end_turn = min(prior["end_turn"], turn_limit)
    remaining_min = _eta_integrate(tuple(rates), current_turn, end_turn)
    remaining_min *= prior["stall_mult"]
    eta_h = remaining_min / 60

    # Confidence interval
    sigma = 0.25
    lo_rates = tuple(r * (1 - sigma) for r in rates)
    hi_rates = tuple(r * (1 + sigma) for r in rates)
    lo_end = max(current_turn + 1, min(end_turn, 250))
    eta_lo = _eta_integrate(lo_rates, current_turn, lo_end) / 60
    eta_hi = (
        _eta_integrate(hi_rates, current_turn, turn_limit) * prior["stall_mult"] / 60
    )

    return {
        "eta_h": round(eta_h, 1),
        "lo_h": round(eta_lo, 1),
        "hi_h": round(eta_hi, 1),
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        log.error("Config not found: %s", CONFIG_PATH)
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Machine abstraction
# ---------------------------------------------------------------------------


@dataclass
class Machine:
    name: str
    ssh_target: str
    os: str  # "windows" | "linux" | "macos"
    repo: str
    display_env: dict[str, str] = field(default_factory=dict)
    ssh_timeout: int = 20

    def ssh(self, cmd: str, timeout: int | None = None) -> tuple[int, str]:
        """Run command via SSH. Returns (returncode, stdout)."""
        t = timeout or self.ssh_timeout
        try:
            r = subprocess.run(
                [
                    "ssh",
                    "-o",
                    f"ConnectTimeout={t}",
                    "-o",
                    "BatchMode=yes",
                    self.ssh_target,
                    cmd,
                ],
                capture_output=True,
                text=True,
                timeout=t + 10,
            )
            return r.returncode, r.stdout.strip()
        except subprocess.TimeoutExpired:
            return -1, "SSH_TIMEOUT"
        except Exception as e:
            return -1, str(e)

    def is_reachable(self) -> bool:
        rc, _ = self.ssh("echo ok", timeout=10)
        return rc == 0

    def get_version(self) -> str:
        if self.os == "windows":
            cmd = f"cd {self.repo} && git describe --tags --always 2>nul"
        else:
            cmd = f"cd {self.repo} && git describe --tags --always"
        rc, out = self.ssh(cmd)
        return out if rc == 0 else "UNKNOWN"

    def is_game_running(self) -> bool:
        if self.os == "windows":
            rc, out = self.ssh(
                'tasklist /FI "IMAGENAME eq CivilizationVI_DX12.exe" /NH 2>nul '
                "| findstr /I Civ >nul && echo YES || echo NO"
            )
        else:
            rc, out = self.ssh(
                "pgrep -x Civ6Sub >/dev/null 2>&1 && echo YES || echo NO"
            )
        return "YES" in out

    def is_runner_running(self) -> bool:
        if self.os == "windows":
            # Match runner.py specifically, not any python.exe.
            # Use EncodedCommand to avoid quote-mangling through SSH→cmd→PowerShell.
            ps = (
                "if (Get-Process python -ErrorAction SilentlyContinue "
                "| Where-Object { $_.CommandLine -match 'runner.py' }) "
                "{ 'YES' } else { 'NO' }"
            )
            encoded = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
            rc, out = self.ssh(
                f"powershell -EncodedCommand {encoded}",
                timeout=15,
            )
        else:
            rc, out = self.ssh(
                'pgrep -f "evals/runner.py\\|inspect eval" >/dev/null 2>&1 '
                "&& echo YES || echo NO"
            )
        return "YES" in out

    def check_completed(self) -> bool:
        """Check if runner left a completion sentinel."""
        if self.os == "windows":
            rc, out = self.ssh(
                "if exist %USERPROFILE%\\civbench_done (echo YES) else (echo NO)",
                timeout=10,
            )
        else:
            rc, out = self.ssh(
                "test -f ~/civbench_done && echo YES || echo NO", timeout=10
            )
        return "YES" in out

    def clear_completion_sentinel(self) -> None:
        if self.os == "windows":
            self.ssh("del %USERPROFILE%\\civbench_done 2>nul", timeout=5)
        else:
            self.ssh("rm -f ~/civbench_done", timeout=5)

    def get_latest_turn(self) -> int | None:
        """Parse the last diary entry for turn number."""
        if self.os == "windows":
            diary_dir = "%USERPROFILE%\\.civ6-mcp"
            # Hacky but works — find newest diary, read last line
            cmd = (
                f'powershell -Command "'
                f"$f = Get-ChildItem '{diary_dir}\\diary_*.jsonl' -Exclude '*cities*' "
                f"| Sort-Object LastWriteTime -Descending | Select-Object -First 1; "
                f"if ($f) {{ (Get-Content $f.FullName -Tail 1 | ConvertFrom-Json).turn }}"
                f'"'
            )
        else:
            cmd = (
                "ls -t ~/.civ6-mcp/diary_*.jsonl 2>/dev/null | grep -v cities | head -1 "
                "| xargs -I{} tail -1 {} 2>/dev/null "
                '| python3 -c \'import sys,json; print(json.load(sys.stdin).get("turn",""))\' 2>/dev/null'
            )
        rc, out = self.ssh(cmd, timeout=15)
        try:
            return int(out.strip()) if out.strip().isdigit() else None
        except (ValueError, AttributeError):
            return None

    def read_heartbeat(self) -> dict | None:
        """Read heartbeat.json from remote machine. Returns parsed dict or None."""
        if self.os == "windows":
            cmd = "type %USERPROFILE%\\.civ6-mcp\\heartbeat.json 2>nul"
        else:
            cmd = "cat ~/.civ6-mcp/heartbeat.json 2>/dev/null"
        rc, out = self.ssh(cmd, timeout=10)
        if rc != 0 or not out.strip():
            return None
        try:
            return json.loads(out.strip())
        except json.JSONDecodeError:
            return None

    def kill_game(self) -> None:
        if self.os == "windows":
            self.ssh("taskkill /F /IM CivilizationVI_DX12.exe 2>nul", timeout=10)
            self.ssh("taskkill /F /IM CivilizationVI.exe 2>nul", timeout=10)
        else:
            self.ssh("killall -9 Civ6Sub Civ6 2>/dev/null", timeout=10)

    def kill_runner(self) -> None:
        # Try heartbeat PID first (most reliable across all platforms)
        hb = self.read_heartbeat()
        hb_pid = hb.get("pid") if hb else None
        if hb_pid and str(hb_pid).isdigit():
            if self.os == "windows":
                self.ssh(f"taskkill /F /PID {hb_pid}", timeout=10)
            else:
                self.ssh(f"kill -9 {hb_pid} 2>/dev/null", timeout=10)

        if self.os == "windows":
            # Fallback: find runner PID via PowerShell CommandLine match
            ps = (
                "Get-Process python -ErrorAction SilentlyContinue "
                "| Where-Object { $_.CommandLine -match 'runner' } "
                "| ForEach-Object { $_.Id }"
            )
            encoded = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
            rc, out = self.ssh(f"powershell -EncodedCommand {encoded}", timeout=15)
            for pid in out.strip().split():
                if pid.strip().isdigit():
                    self.ssh(f"taskkill /F /PID {pid.strip()}", timeout=10)
            self.ssh("schtasks /End /TN CivBench 2>nul", timeout=10)
        else:
            self.ssh(
                "tmux kill-session -t civbench 2>/dev/null; pkill -f runner.py 2>/dev/null",
                timeout=10,
            )

    def clean_autosaves(self) -> str:
        """Remove MCP autosaves AND game autosaves to prevent wrong-save loads."""
        if self.os == "windows":
            save_dir = "C:\\Users\\%USERNAME%\\Documents\\My Games\\Sid Meier's Civilization VI\\Saves\\Single"
            self.ssh(f'del /Q "{save_dir}\\0_MCP_*.Civ6Save" 2>nul')
            # Also clean game autosaves that "Continue Game" would load
            self.ssh(f'del /Q "{save_dir}\\auto\\AutoSave_*.Civ6Save" 2>nul')
            rc, out = self.ssh("echo CLEANED")
        else:
            # Use Sid* glob to avoid apostrophe escaping issues through SSH
            self.ssh(
                "rm -f ~/.local/share/aspyr-media/Sid*/Saves/Single/0_MCP_*.Civ6Save "
                "2>/dev/null"
            )
            # Also clean game autosaves that "Continue Game" would load
            self.ssh(
                "rm -f ~/.local/share/aspyr-media/Sid*/Saves/Single/auto/AutoSave_*.Civ6Save "
                "2>/dev/null"
            )
            rc, out = self.ssh("echo CLEANED")
        return out

    def launch_runner(self, model: str, scenario: str, runs: int = 1) -> bool:
        """Launch the benchmark runner. Returns True if launch command succeeded."""
        if self.os == "windows":
            # Write batch file via PowerShell base64 to avoid all escaping issues
            bat_content = (
                f"@echo off\r\n"
                f"cd /d {self.repo}\r\n"
                f".venv\\Scripts\\python.exe -u evals/runner.py "
                f"--model {model} --scenarios {scenario} --runs {runs} "
                f">> %USERPROFILE%\\civbench_run.log 2>&1\r\n"
                f"if %ERRORLEVEL% EQU 0 echo DONE > %USERPROFILE%\\civbench_done\r\n"
            )
            encoded = base64.b64encode(bat_content.encode("utf-8")).decode("ascii")
            self.ssh(
                f'powershell -Command "[System.Text.Encoding]::UTF8.GetString('
                f"[System.Convert]::FromBase64String('{encoded}'))"
                f" | Set-Content -Path '{self.repo}\\run_bench.bat' -NoNewline\""
            )
            self.ssh(
                'schtasks /Create /TN "CivBench" /TR '
                f'"{self.repo}\\run_bench.bat" /SC ONCE /ST 00:00 /RL HIGHEST /F'
            )
            rc, out = self.ssh('schtasks /Run /TN "CivBench"')
            return "SUCCESS" in out or rc == 0
        else:
            env_parts = [f"export {k}={v};" for k, v in self.display_env.items()]
            env_str = " ".join(env_parts)
            # Kill any existing civbench tmux session before creating a new one.
            # Prevents "duplicate session" errors on retry/relaunch.
            self.ssh("tmux kill-session -t civbench 2>/dev/null", timeout=5)
            rc, out = self.ssh(
                f"{env_str} tmux new-session -d -s civbench "
                f'"cd {self.repo} && uv run python -u evals/runner.py '
                f"--model {model} --scenarios {scenario} --runs {runs} "
                f'2>&1 | tee ~/civbench_run.log && touch ~/civbench_done"'
            )
            return rc == 0

    def discover_run_id(self) -> str | None:
        """Extract run_id from the most recent diary file on this machine."""
        if self.os == "windows":
            # Get basename of newest diary file, extract last _-delimited segment
            ps = (
                "$f = Get-ChildItem $env:USERPROFILE\\.civ6-mcp\\diary_*.jsonl "
                "-Exclude '*cities*' -ErrorAction SilentlyContinue "
                "| Sort-Object LastWriteTime -Descending | Select-Object -First 1; "
                "if ($f) { $f.BaseName.Split('_')[-1] }"
            )
            encoded = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
            cmd = f"powershell -EncodedCommand {encoded}"
        else:
            cmd = (
                "ls -t ~/.civ6-mcp/diary_*.jsonl 2>/dev/null | grep -v cities | head -1 "
                "| xargs -I{} basename {} .jsonl | rev | cut -d_ -f1 | rev"
            )
        rc, out = self.ssh(cmd, timeout=15)
        rid = out.strip().split("\n")[-1].strip()  # take last line only
        return rid if rc == 0 and rid and len(rid) > 2 else None

    def sync_to_convex(self) -> bool:
        """Run convex_sync.py --upload on this machine. Returns True on success."""
        if self.os == "windows":
            diary_dir = "%USERPROFILE%\\.civ6-mcp"
            cmd = (
                f"cd /d {self.repo} && "
                f".venv\\Scripts\\python.exe scripts/convex_sync.py "
                f"--upload {diary_dir} --prod"
            )
        else:
            cmd = (
                f"cd {self.repo} && "
                f"uv run python scripts/convex_sync.py "
                f"--upload ~/.civ6-mcp --prod"
            )
        rc, out = self.ssh(cmd, timeout=300)
        if rc != 0:
            log.warning("Sync failed on %s: %s", self.name, out[:200])
        return rc == 0

    def clear_local_telemetry(self) -> str:
        """Remove all local diary/log/spatial files. Azure has backups."""
        if self.os == "windows":
            rc, out = self.ssh(
                'powershell -Command "Remove-Item $env:USERPROFILE\\.civ6-mcp\\* -Force -ErrorAction SilentlyContinue; echo CLEARED"',
                timeout=15,
            )
        else:
            rc, out = self.ssh("rm -rf ~/.civ6-mcp/* && echo CLEARED", timeout=10)
        return out

    def tail_log(self, n: int = 10) -> str:
        if self.os == "windows":
            rc, out = self.ssh(
                f'powershell -Command "Get-Content %USERPROFILE%\\civbench_run.log -Tail {n}"',
                timeout=15,
            )
        else:
            rc, out = self.ssh(f"tail -{n} ~/civbench_run.log 2>/dev/null", timeout=10)
        return out


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    machine_name: str
    model: str
    scenario: str
    run_num: int
    status: str = "pending"  # pending, launching, running, completing, done, failed
    run_id: str | None = None
    started_at: float = 0
    finished_at: float = 0
    last_turn: int = 0
    last_turn_change: float = 0
    retries: int = 0
    fail_reason: str = ""
    synced: bool = False
    score: int = 0
    outcome: str = ""  # "victory", "defeat", ""
    boot_phase: str = ""  # last known heartbeat phase

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"jobs": {}}


def save_state(state: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


def alert(message: str) -> None:
    """Send ntfy alert + log."""
    log.warning("ALERT: %s", message)
    config = load_config()
    # Try to read webhook from config or environment
    webhook = config.get("defaults", {}).get("alert_webhook", "")
    if not webhook:
        return
    try:
        import urllib.request

        payload = json.dumps(
            {
                "message": message,
                "title": "CivBench Orchestrator",
                "priority": 4,
                "tags": ["warning"],
            }
        ).encode()
        req = urllib.request.Request(
            webhook, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_preflight(
    machines: dict[str, Machine], machine_names: list[str] | None
) -> bool:
    """Validate all machines are ready. Returns True if all pass."""
    targets = {
        n: machines[n] for n in (machine_names or machines.keys()) if n in machines
    }
    all_ok = True

    for name, m in targets.items():
        print(f"\n  {name} ({m.os}, {m.ssh_target})")
        print(f"  {'─' * 40}")

        # Reachable
        if not m.is_reachable():
            print("    ✗ OFFLINE")
            all_ok = False
            continue
        print("    ✓ Reachable")

        # Version
        version = m.get_version()
        print(f"    {'✓' if version != 'UNKNOWN' else '✗'} Version: {version}")
        if version == "UNKNOWN":
            all_ok = False

        # Game running
        game = m.is_game_running()
        runner = m.is_runner_running()
        print(
            f"    {'●' if game else '○'} Civ VI: {'running' if game else 'not running'}"
        )
        print(f"    {'●' if runner else '○'} Runner: {'active' if runner else 'idle'}")

        # Save files
        if m.os == "windows":
            rc, out = m.ssh(
                f"dir {m.repo}\\evals\\saves\\0A_GROUND_CONTROL.Civ6Save 2>nul && echo FOUND || echo MISSING"
            )
        else:
            rc, out = m.ssh(
                f"test -f {m.repo}/evals/saves/0A_GROUND_CONTROL.Civ6Save && echo FOUND || echo MISSING"
            )
        has_saves = "FOUND" in out
        print(
            f"    {'✓' if has_saves else '✗'} Saves: {'present' if has_saves else 'MISSING'}"
        )
        if not has_saves:
            all_ok = False

    print()
    return all_ok


def cmd_status(machines: dict[str, Machine]) -> None:
    """Show fleet status dashboard."""
    state = load_state()
    jobs = state.get("jobs", {})

    print("\n╔═══════════════════════════════════════════════════╗")
    print(f"║  CivBench Orchestrator     {time.strftime('%H:%M %b %d %Y')}  ║")
    print("╚═══════════════════════════════════════════════════╝")

    print("\nFleet")
    print(f"  {'─' * 50}")
    for name, m in machines.items():
        reachable = m.is_reachable()
        if not reachable:
            print(f"  {name:<12} OFFLINE")
            continue
        version = m.get_version()
        game = m.is_game_running()
        hb = m.read_heartbeat()
        if hb:
            hb_age = time.time() - hb.get("ts", 0)
            phase = hb.get("phase", "?")
            turn = hb.get("turn", 0)
            alive = hb_age < 90
            print(
                f"  {name:<12} {version:<20} "
                f"{'CIV' if game else '   '} "
                f"{'HB' if alive else '  '} "
                f"{phase:<10} T{turn}"
            )
        else:
            runner = m.is_runner_running()
            turn = m.get_latest_turn()
            turn_str = f"T{turn}" if turn is not None else ""
            print(
                f"  {name:<12} {version:<20} "
                f"{'CIV' if game else '   '} "
                f"{'RUN' if runner else '   '} "
                f"{'':10} {turn_str}"
            )

    active_jobs = {
        k: v
        for k, v in jobs.items()
        if v.get("status") in ("launching", "running", "completing")
    }
    pending_jobs = {k: v for k, v in jobs.items() if v.get("status") == "pending"}
    done_jobs = {k: v for k, v in jobs.items() if v.get("status") == "done"}
    failed_jobs = {k: v for k, v in jobs.items() if v.get("status") == "failed"}

    if active_jobs:
        print("\nActive Jobs")
        print(f"  {'─' * 60}")
        for jid, j in active_jobs.items():
            sa = j.get("started_at", 0)
            elapsed_h = (time.time() - sa) / 3600 if sa > 0 else 0
            turn = j.get("last_turn", 0)
            model_short = j.get("model", "?").rsplit("/", 1)[-1]
            # Calculate rate and ETA (non-linear model)
            rate_str = ""
            eta_str = ""
            if turn > 5 and elapsed_h > 0.01:
                rate_str = f"{elapsed_h * 60 / turn:.1f}m/t"
                est = estimate_eta(j.get("model", ""), turn, elapsed_h)
                eta_str = f"~{est['eta_h']}h [{est['lo_h']}-{est['hi_h']}h]"
            rid = j.get("run_id") or ""
            print(
                f"  {j.get('machine_name', '?'):<10} {model_short:<18} "
                f"T{turn:>3}  {elapsed_h:.1f}h  {rate_str:>7}  {eta_str:>10}  {rid}"
            )

    if done_jobs:
        print(f"\nCompleted ({len(done_jobs)})")
        print(f"  {'─' * 60}")
        for jid, j in done_jobs.items():
            model_short = j.get("model", "?").rsplit("/", 1)[-1]
            elapsed = (
                (j.get("finished_at", 0) - j.get("started_at", 0)) / 3600
                if j.get("finished_at")
                else 0
            )
            synced = "synced" if j.get("synced") else "pending sync"
            rid = j.get("run_id") or "?"
            print(
                f"  ✓ {rid:<25} {model_short:<18} T{j.get('last_turn', '?'):>3}  {elapsed:.1f}h  {synced}"
            )

    if pending_jobs:
        print(f"\nPending: {len(pending_jobs)} jobs")
    if failed_jobs:
        print(f"\nFailed ({len(failed_jobs)})")
        for jid, j in failed_jobs.items():
            print(f"  ✗ {jid}: {j.get('fail_reason', '?')}")
    print()


def cmd_kill_all(machines: dict[str, Machine]) -> None:
    """Kill all runners and games on all machines."""
    for name, m in machines.items():
        print(f"  Killing {name}... ", end="", flush=True)
        if not m.is_reachable():
            print("OFFLINE")
            continue
        m.kill_runner()
        m.kill_game()
        print("done")

    # Update state
    state = load_state()
    for jid, j in state.get("jobs", {}).items():
        if j.get("status") in ("launching", "running", "completing"):
            j["status"] = "failed"
            j["fail_reason"] = "killed by operator"
    save_state(state)
    print("  All jobs marked as failed.")


def cmd_launch(
    machines: dict[str, Machine],
    machine_names: list[str],
    models: list[str],
    scenarios: list[str],
    runs: int,
    config: dict[str, Any],
) -> None:
    """Launch benchmark runs and monitor until completion."""
    defaults = config.get("defaults", {})
    poll_interval = defaults.get("poll_interval", 30)
    stall_alert_min = defaults.get("stall_alert_minutes", 30)
    stall_kill_min = defaults.get("stall_kill_minutes", 60)
    max_retries = defaults.get("max_retries", 2)

    # Clean slate: clear stale state, sentinels, and autosaves on ALL machines
    log.info("Clearing stale state and sentinels on all machines...")
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    for name in machine_names:
        m = machines.get(name)
        if m and m.is_reachable():
            m.clear_completion_sentinel()
            m.clean_autosaves()
            m.clear_local_telemetry()
            log.info("  %s: cleaned", name)

    # Build job queue: interleave models across machines so each machine
    # runs a DIFFERENT model in each round. Round 1: machine[0]→model[0],
    # machine[1]→model[1], machine[2]→model[2]. Round 2: same assignment.
    jobs: dict[str, Job] = {}
    machine_list = [machines[n] for n in machine_names if n in machines]
    for scenario in scenarios:
        for run_num in range(1, runs + 1):
            for m_idx, model in enumerate(models):
                machine = machine_list[m_idx % len(machine_list)]
                jid = f"{machine.name}_{model.rsplit('/', 1)[-1]}_{scenario}_{run_num}"
                jobs[jid] = Job(
                    id=jid,
                    machine_name=machine.name,
                    model=model,
                    scenario=scenario,
                    run_num=run_num,
                )

    # Persist initial state
    state = {
        "started_at": time.time(),
        "jobs": {jid: j.to_dict() for jid, j in jobs.items()},
    }
    save_state(state)

    print(f"\nScheduled {len(jobs)} jobs across {len(machine_list)} machines:")
    for jid, j in jobs.items():
        print(
            f"  {j.machine_name:<12} {j.model.rsplit('/', 1)[-1]:<25} {j.scenario:<20} run {j.run_num}"
        )
    print()

    # Track which machine is busy
    machine_jobs: dict[str, str | None] = {n: None for n in machine_names}
    poll_count = 0

    while True:
        # Dispatch pending jobs to idle machines
        for jid, job in jobs.items():
            if job.status != "pending":
                continue
            m_name = job.machine_name
            if machine_jobs.get(m_name) is not None:
                continue  # machine busy

            m = machines[m_name]
            log.info("Launching %s on %s", jid, m_name)
            job.status = "launching"

            # Pre-flight
            if not m.is_reachable():
                job.status = "failed"
                job.fail_reason = "machine unreachable"
                alert(f"Cannot launch {jid}: {m_name} unreachable")
                continue

            # Clean autosaves + stale completion sentinel
            m.clean_autosaves()
            m.clear_completion_sentinel()

            # Kill any stale processes
            if m.is_runner_running():
                m.kill_runner()
                time.sleep(3)

            # Launch
            ok = m.launch_runner(job.model, job.scenario, 1)  # 1 run at a time
            if not ok:
                job.status = "failed"
                job.fail_reason = "launch command failed"
                alert(f"Launch failed: {jid}")
                continue

            job.status = "running"
            job.started_at = time.time()
            job.last_turn_change = time.time()
            machine_jobs[m_name] = jid
            log.info("Running: %s", jid)

        # Poll active jobs — heartbeat-first health detection
        for jid, job in jobs.items():
            if job.status != "running":
                continue

            m = machines[job.machine_name]
            launch_age = time.time() - job.started_at
            hb = m.read_heartbeat()

            if hb is not None:
                phase = hb.get("phase", "")
                hb_turn = hb.get("turn", 0)
                hb_ts = hb.get("ts", 0)
                age = time.time() - hb_ts

                job.boot_phase = phase

                # Bind run_id from heartbeat if we don't have it yet
                if not job.run_id and hb.get("run_id"):
                    job.run_id = hb["run_id"]

                if phase in ("error", "finished"):
                    # Will be caught by completion sentinel check below
                    pass
                elif age > 90:
                    # Heartbeat stale — runner probably dead. Confirm.
                    time.sleep(5)
                    hb2 = m.read_heartbeat()
                    hb2_ts = hb2.get("ts", 0) if hb2 else 0
                    if time.time() - hb2_ts > 90:
                        log.warning(
                            "Heartbeat stale: %s (phase=%s, T%d, age=%.0fs)",
                            jid,
                            phase,
                            hb_turn,
                            age,
                        )
                        if job.retries < max_retries:
                            job.retries += 1
                            job.status = "pending"
                            machine_jobs[job.machine_name] = None
                            alert(
                                f"Runner died: {jid} (heartbeat stale, {phase}:T{hb_turn}) "
                                f"— retry {job.retries}/{max_retries}"
                            )
                        else:
                            job.status = "failed"
                            job.fail_reason = (
                                f"heartbeat stale (phase={phase}, T{hb_turn})"
                            )
                            machine_jobs[job.machine_name] = None
                            alert(f"Job failed: {jid} — heartbeat stale")
                        continue
                else:
                    # Heartbeat fresh — update turn
                    if hb_turn > 0 and hb_turn != job.last_turn:
                        old = job.last_turn
                        job.last_turn = hb_turn
                        job.last_turn_change = time.time()
                        log.info("Turn advance: %s T%d->T%d", jid, old, hb_turn)

                # Stall detection (using heartbeat turn)
                if job.last_turn > 0:
                    stall_min = (time.time() - job.last_turn_change) / 60
                    if stall_min > stall_kill_min:
                        log.error(
                            "Stall timeout: %s at T%d for %.0fm",
                            jid,
                            job.last_turn,
                            stall_min,
                        )
                        m.kill_runner()
                        m.kill_game()
                        if job.retries < max_retries:
                            job.retries += 1
                            job.status = "pending"
                            machine_jobs[job.machine_name] = None
                            alert(
                                f"Stall kill: {jid} T{job.last_turn} "
                                f"({stall_min:.0f}m) — retry {job.retries}"
                            )
                        else:
                            job.status = "failed"
                            job.fail_reason = (
                                f"stall at T{job.last_turn} for {stall_min:.0f}m"
                            )
                            machine_jobs[job.machine_name] = None
                            alert(f"Job failed: {jid} — stall timeout")
                    elif stall_min > stall_alert_min:
                        log.warning(
                            "Stall: %s at T%d for %.0fm",
                            jid,
                            job.last_turn,
                            stall_min,
                        )
                elif launch_age > 600:
                    # Boot timeout: 10 min with heartbeat but never reached T>0
                    log.error(
                        "Boot timeout: %s stuck at phase=%s for %.0fm",
                        jid,
                        phase,
                        launch_age / 60,
                    )
                    m.kill_runner()
                    m.kill_game()
                    if job.retries < max_retries:
                        job.retries += 1
                        job.status = "pending"
                        machine_jobs[job.machine_name] = None
                        alert(f"Boot timeout: {jid} ({phase}) — retry {job.retries}")
                    else:
                        job.status = "failed"
                        job.fail_reason = f"boot timeout (phase={phase})"
                        machine_jobs[job.machine_name] = None
                        alert(f"Job failed: {jid} — boot timeout")
            else:
                # No heartbeat — fall back to process detection + diary
                if launch_age < 180:
                    continue  # grace period
                if not m.is_runner_running():
                    time.sleep(5)
                    if not m.is_runner_running():
                        if job.retries < max_retries:
                            job.retries += 1
                            job.status = "pending"
                            machine_jobs[job.machine_name] = None
                            log.warning(
                                "Runner died (no heartbeat): %s, retry %d/%d",
                                jid,
                                job.retries,
                                max_retries,
                            )
                            alert(
                                f"Runner died: {jid} — retry {job.retries}/{max_retries}"
                            )
                        else:
                            job.status = "failed"
                            job.fail_reason = "runner died after max retries"
                            machine_jobs[job.machine_name] = None
                            alert(f"Job failed: {jid} — runner died")
                        continue

                # Fall back to diary for turn
                turn = m.get_latest_turn()
                if turn is not None and turn != job.last_turn:
                    old = job.last_turn
                    job.last_turn = turn
                    job.last_turn_change = time.time()
                    log.info("Turn advance (diary): %s T%d->T%d", jid, old, turn)

            # Boot tail: if stuck at T0 beyond grace, surface runner log
            if (
                job.last_turn == 0
                and launch_age > 300
                and poll_count % 5 == 0
                and job.status == "running"
            ):
                tail = m.tail_log(3)
                if tail:
                    log.info("Boot tail %s: %s", jid, tail.replace("\n", " | ")[:200])

        # Check for completed jobs via sentinel file → run post-game pipeline
        # (skip during grace period — runner may not have started yet)
        for jid, job in jobs.items():
            if job.status != "running":
                continue
            if time.time() - job.started_at < 180:
                continue  # grace period — game still loading
            m = machines[job.machine_name]
            if m.check_completed():
                job.status = "completing"
                job.finished_at = time.time()
                m.clear_completion_sentinel()
                # Persist immediately — if orchestrator crashes mid-pipeline,
                # we don't lose the completion or re-trigger the job
                state["jobs"][jid] = job.to_dict()
                save_state(state)
                elapsed = (job.finished_at - job.started_at) / 3600
                log.info(
                    "Game finished: %s T%d in %.1fh — running post-game pipeline",
                    jid,
                    job.last_turn,
                    elapsed,
                )

                # 1. Discover run_id
                run_id = m.discover_run_id()
                if run_id:
                    job.run_id = run_id
                    log.info("  Run ID: %s", run_id)

                # 2. Sync to Convex
                log.info("  Syncing to Convex...")
                if m.sync_to_convex():
                    job.synced = True
                    log.info("  Sync OK")
                else:
                    log.warning(
                        "  Sync failed — data still on machine, can retry with 'sync' command"
                    )

                # 3. Rich alert
                short_model = job.model.rsplit("/", 1)[-1]
                rid_str = job.run_id or "?"
                alert(
                    f"✓ {rid_str} | {short_model} | {job.scenario} | "
                    f"T{job.last_turn} | {elapsed:.1f}h"
                )

                # 4. Mark done, free machine
                job.status = "done"
                machine_jobs[job.machine_name] = None
                log.info("Completed: %s", jid)

        # Persist state
        state = {
            "started_at": state.get("started_at", time.time()),
            "jobs": {jid: j.to_dict() for jid, j in jobs.items()},
        }
        save_state(state)

        # Check if all done
        statuses = [j.status for j in jobs.values()]
        if all(s in ("done", "failed") for s in statuses):
            done_count = statuses.count("done")
            fail_count = statuses.count("failed")
            print(f"\n{'═' * 50}")
            print(f"  All jobs complete: {done_count} done, {fail_count} failed")
            print(f"{'═' * 50}")
            alert(f"CivBench batch complete: {done_count} done, {fail_count} failed")
            break

        # Print compact status line (with phase info)
        running = sum(1 for j in jobs.values() if j.status == "running")
        pending = sum(1 for j in jobs.values() if j.status == "pending")
        done = sum(1 for j in jobs.values() if j.status == "done")
        failed = sum(1 for j in jobs.values() if j.status == "failed")
        active_info = " | ".join(
            f"{j.machine_name}:{j.boot_phase or '?'}:T{j.last_turn}"
            for j in jobs.values()
            if j.status == "running"
        )
        sys.stdout.write(
            f"\r  [{done}ok {running}run {pending}wait {failed}fail] {active_info}    "
        )
        sys.stdout.flush()

        # Periodic full status to log file (every ~5 min)
        poll_count += 1
        if poll_count % 10 == 0:
            status_parts = []
            for j in jobs.values():
                if j.status == "running":
                    elapsed_h = (time.time() - j.started_at) / 3600
                    status_parts.append(
                        f"{j.machine_name}:{j.boot_phase}:T{j.last_turn}:{elapsed_h:.1f}h"
                    )
            log.info(
                "STATUS: [%d done %d run %d pend %d fail] %s",
                done,
                running,
                pending,
                failed,
                " | ".join(status_parts),
            )

        last_poll = time.time()
        time.sleep(poll_interval)

        # Detect sleep/suspend: if wall clock jumped far beyond poll interval,
        # the host was asleep. Reset stall timers so we don't kill healthy jobs.
        wake_gap = time.time() - last_poll - poll_interval
        if wake_gap > poll_interval * 2:
            log.warning(
                "Time jump detected (%.0fs gap) — likely sleep/suspend. "
                "Resetting stall timers for all running jobs.",
                wake_gap + poll_interval,
            )
            for job in jobs.values():
                if job.status == "running":
                    job.last_turn_change = time.time()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Persistent log file for multi-day runs
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(CONFIG_DIR / "orchestrator.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("orchestrator").addHandler(fh)

    parser = argparse.ArgumentParser(description="CivBench Orchestrator")
    sub = parser.add_subparsers(dest="command")

    # preflight
    p_pre = sub.add_parser("preflight", help="Validate machines are ready")
    p_pre.add_argument("--machines", help="Comma-separated machine names")

    # launch
    p_launch = sub.add_parser("launch", help="Launch benchmark runs")
    p_launch.add_argument(
        "--scenarios", required=True, help="Comma-separated scenario IDs"
    )
    p_launch.add_argument(
        "--models", required=True, help="Comma-separated model aliases or full IDs"
    )
    p_launch.add_argument(
        "--runs", type=int, default=3, help="Runs per (model, scenario)"
    )
    p_launch.add_argument(
        "--machines", required=True, help="Comma-separated machine names"
    )

    # status
    sub.add_parser("status", help="Show fleet status + active job details")

    # kill-all
    sub.add_parser("kill-all", help="Kill all runners and games")

    # sync
    p_sync = sub.add_parser("sync", help="Sync telemetry to Convex")
    p_sync.add_argument(
        "--machines", help="Comma-separated machine names (default: all)"
    )

    # summary
    sub.add_parser("summary", help="Aggregate results by model and scenario")

    # logs
    p_logs = sub.add_parser("logs", help="Tail remote runner logs")
    p_logs.add_argument("--machine", required=True, help="Machine name")
    p_logs.add_argument(
        "--last", type=int, default=30, help="Number of lines (default: 30)"
    )
    p_logs.add_argument(
        "--errors", action="store_true", help="Show only error/warning lines"
    )

    # resume
    sub.add_parser("resume", help="Resume from saved state")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    config = load_config()
    machine_defs = config.get("machines", {})
    defaults = config.get("defaults", {})
    aliases = config.get("model_aliases", {})

    # Build Machine objects
    machines: dict[str, Machine] = {}
    for name, mdef in machine_defs.items():
        machines[name] = Machine(
            name=name,
            ssh_target=mdef["ssh"],
            os=mdef["os"],
            repo=mdef["repo"],
            display_env=mdef.get("display_env", {}),
            ssh_timeout=defaults.get("ssh_timeout", 20),
        )

    if args.command == "preflight":
        names = args.machines.split(",") if args.machines else None
        ok = cmd_preflight(machines, names)
        sys.exit(0 if ok else 1)

    elif args.command == "status":
        cmd_status(machines)

    elif args.command == "kill-all":
        cmd_kill_all(machines)

    elif args.command == "launch":
        machine_names = [n.strip() for n in args.machines.split(",")]
        unknown = [n for n in machine_names if n not in machines]
        if unknown:
            log.error(
                "Unknown machines: %s (available: %s)", unknown, list(machines.keys())
            )
            sys.exit(1)
        scenario_list = [s.strip() for s in args.scenarios.split(",")]
        model_list = [aliases.get(m.strip(), m.strip()) for m in args.models.split(",")]
        cmd_launch(
            machines, machine_names, model_list, scenario_list, args.runs, config
        )

    elif args.command == "sync":
        names = (
            [n.strip() for n in args.machines.split(",")]
            if args.machines
            else list(machines.keys())
        )
        for name in names:
            if name not in machines:
                log.warning("Unknown machine: %s", name)
                continue
            m = machines[name]
            print(f"  Syncing {name}... ", end="", flush=True)
            if not m.is_reachable():
                print("OFFLINE")
                continue
            if m.sync_to_convex():
                print("OK")
            else:
                print("FAILED")

    elif args.command == "summary":
        state = load_state()
        jobs_data = state.get("jobs", {})
        if not jobs_data:
            print("No jobs in state.")
        else:
            # Group by scenario, then model
            by_scenario: dict[str, dict[str, list]] = defaultdict(
                lambda: defaultdict(list)
            )
            for jid, j in jobs_data.items():
                by_scenario[j.get("scenario", "?")][
                    j.get("model", "?").rsplit("/", 1)[-1]
                ].append(j)

            for scenario, models_dict in sorted(by_scenario.items()):
                total = sum(len(jl) for jl in models_dict.values())
                done = sum(
                    1
                    for jl in models_dict.values()
                    for j in jl
                    if j.get("status") == "done"
                )
                print(f"\n{scenario} ({done}/{total} done)")
                print(f"  {'─' * 55}")
                for model, jlist in sorted(models_dict.items()):
                    n = len(jlist)
                    d = sum(1 for j in jlist if j.get("status") == "done")
                    turns = [
                        j.get("last_turn", 0)
                        for j in jlist
                        if j.get("status") == "done"
                    ]
                    elapsed = [
                        (j.get("finished_at", 0) - j.get("started_at", 0)) / 3600
                        for j in jlist
                        if j.get("status") == "done" and j.get("finished_at")
                    ]
                    avg_t = f"avg T{sum(turns) // len(turns)}" if turns else ""
                    avg_h = f"avg {sum(elapsed) / len(elapsed):.1f}h" if elapsed else ""
                    rids = [
                        j.get("run_id", "?") for j in jlist if j.get("status") == "done"
                    ]
                    print(
                        f"  {model:<22} {d}/{n}  {avg_t:>8}  {avg_h:>10}  {' '.join(rids)}"
                    )
            print()

    elif args.command == "logs":
        if not args.machine or args.machine not in machines:
            log.error("Specify a valid machine with --machine")
            sys.exit(1)
        m = machines[args.machine]
        n = args.last or 30
        output = m.tail_log(n)
        if args.errors:
            output = "\n".join(
                line
                for line in output.split("\n")
                if any(
                    kw in line.lower()
                    for kw in ("error", "warning", "fail", "traceback", "exception")
                )
            )
        print(output)

    elif args.command == "resume":
        state = load_state()
        if not state.get("jobs"):
            print("No saved state to resume.")
            return
        # Rebuild jobs and relaunch pending/running ones
        print("Resume not yet implemented — use launch with explicit args")


if __name__ == "__main__":
    main()
