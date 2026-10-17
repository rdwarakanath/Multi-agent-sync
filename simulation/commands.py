"""
commands.py -- operator command layer and mission state machine.

    UI / keyboard / CLI  --text-->  CommandManager.execute()
                                         |
                                         v   (public simulation API only)
                                   MultiAgentSimulation

The command layer is STATE based, not a bag of hacks:

    MissionState: IDLE, FOLLOWING, GOTO, SCATTERING, REGROUPING,
                  TASK_EXECUTION, STOPPED

`execute()` validates and applies a command; `update()` is called every
simulation tick and advances the state machine (GOTO arrival, SCATTER /
REGROUP completion, task progress ...).

Commands never move robots directly.  They change targets / modes and send
messages through the network model; the controllers + plant then produce the
motion.  Follower-side changes only happen when the command message is
actually DELIVERED (lost commands are retried by the simulation until ACKed).

Supported commands (case-insensitive)
-------------------------------------
GOTO x y            leader drives to a world coordinate
MOVE dx dy          leader drives to (current position + offset)
SCATTER             followers disperse to individual safe points
REGROUP             followers return to their formation slots
FOLLOW              normal leader-follower behaviour (formation holding)
STOP                stop everything safely            (remembers the mission)
RESUME              continue the interrupted mission
FORMATION [name]    show offsets, or switch to WEDGE | LINE | COLUMN
STATUS              one-screen status summary
TASK id x y [prio]  create a task          ALLOCATE   assign pending tasks
TASKS               list tasks             CLEARTASKS delete all tasks
MODE AUTO|MANUAL|COMMAND   choose how the leader is driven

Single-agent prefix (architecture ready): "AGENT 2 GOTO 10 5", "AGENT 3 STOP".
GOTO / STOP / RESUME are implemented per agent; the rest are swarm-wide.
"""
from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Deque, List, Optional

import numpy as np

from .agent import AgentMode
from .collision import validate_target
from .config import Config
from .formation import (FORMATION_PRESETS, regroup_complete, scatter_targets)
from .messages import (COMMAND, FOLLOW, REGROUP, SCATTER, STOP, TASK_ASSIGN)

if TYPE_CHECKING:                       # avoid a circular import at runtime
    from .simulation import MultiAgentSimulation


class MissionState(str, Enum):
    IDLE = "IDLE"
    FOLLOWING = "FOLLOWING"
    GOTO = "GOTO"
    SCATTERING = "SCATTERING"
    REGROUPING = "REGROUPING"
    TASK_EXECUTION = "TASK_EXECUTION"
    STOPPED = "STOPPED"


class SimMode(str, Enum):
    """How the LEADER is driven (spec section 19)."""
    AUTO = "AUTO"          # predefined trajectory
    MANUAL = "MANUAL"      # keyboard
    COMMAND = "COMMAND"    # operator commands (GOTO ...)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
class CommandError(ValueError):
    """Raised for malformed / unknown commands (never crashes the sim)."""


# name -> (min_args, max_args)
COMMAND_ARITY = {
    "GOTO": (2, 2), "MOVE": (2, 2), "SCATTER": (0, 0), "REGROUP": (0, 0),
    "FOLLOW": (0, 0), "STOP": (0, 0), "RESUME": (0, 0), "FORMATION": (0, 1),
    "STATUS": (0, 0), "TASK": (3, 4), "ALLOCATE": (0, 0), "TASKS": (0, 0),
    "CLEARTASKS": (0, 0), "MODE": (1, 1),
}


@dataclass
class Command:
    name: str
    args: List[str] = field(default_factory=list)
    agent: Optional[int] = None        # None = whole swarm / leader default
    raw: str = ""


@dataclass
class CommandResult:
    ok: bool
    message: str
    command: str = ""


_AGENT_TOKEN = re.compile(r"^(?:AGENT|A|F)(\d+)$", re.I)


def parse_command(text: str) -> Command:
    """Parse operator text into a Command.  Raises CommandError."""
    raw = (text or "").strip()
    if not raw:
        raise CommandError("EMPTY COMMAND")
    tokens = raw.replace(",", " ").split()

    agent: Optional[int] = None
    head = tokens[0].upper()
    m = _AGENT_TOKEN.match(tokens[0])
    if head in ("AGENT", "A") and len(tokens) >= 2 and tokens[1].isdigit():
        agent, tokens = int(tokens[1]), tokens[2:]
    elif m:
        agent, tokens = int(m.group(1)), tokens[1:]
    elif head == "LEADER":
        agent, tokens = 0, tokens[1:]
    elif head == "ALL":
        agent, tokens = None, tokens[1:]
    if not tokens:
        raise CommandError("MISSING COMMAND after agent selector")

    name = tokens[0].upper()
    if name not in COMMAND_ARITY:
        raise CommandError(f"UNKNOWN COMMAND: {tokens[0]}")
    lo, hi = COMMAND_ARITY[name]
    nargs = len(tokens) - 1
    if not lo <= nargs <= hi:
        want = f"{lo}" if lo == hi else f"{lo}-{hi}"
        raise CommandError(f"{name} expects {want} argument(s), got {nargs}")
    return Command(name, tokens[1:], agent, raw)


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        raise CommandError(f"'{s}' is not a number")


# ---------------------------------------------------------------------------
# Command manager / mission state machine
# ---------------------------------------------------------------------------
class CommandManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = MissionState.FOLLOWING
        self.history: Deque[dict] = deque(maxlen=cfg.history_size)  # newest first
        self.scatter_targets: dict = {}
        self._before_goto = MissionState.FOLLOWING
        self._before_stop = MissionState.FOLLOWING
        self._flags: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, text: str, sim: "MultiAgentSimulation") -> CommandResult:
        """Validate + apply one command.  Never raises."""
        raw = (text or "").strip()
        try:
            cmd = parse_command(raw)
            handler = getattr(self, "_cmd_" + cmd.name.lower())
            res = handler(cmd, sim)
        except CommandError as e:
            res = CommandResult(False, str(e), raw)
        except Exception as e:                      # defensive: UI must not crash
            res = CommandResult(False, f"COMMAND FAILED: {type(e).__name__}: {e}", raw)
        res.command = raw
        self._record(sim, res)
        return res

    def history_list(self, n: int = 30) -> List[dict]:
        return list(self.history)[:n]

    # ------------------------------------------------------------------
    # State machine (called once per simulation tick)
    # ------------------------------------------------------------------
    def update(self, sim: "MultiAgentSimulation") -> None:
        cfg = self.cfg
        followers = sim.followers

        # Leader GOTO arrival (independent of mission state)
        if sim.leader.mode == AgentMode.GOTO and sim.leader_reached_goal():
            sim.finish_leader_goto()
            sim.log_status("GOTO COMPLETE")
            if self.state == MissionState.GOTO:
                self.state = self._before_goto

        # GOTO cancelled by something else (keyboard / MODE change)
        if self.state == MissionState.GOTO and sim.leader.mode != AgentMode.GOTO:
            self.state = self._before_goto

        st = self.state
        if st == MissionState.SCATTERING and "scatter_done" not in self._flags:
            if followers and all(
                    f.mode == AgentMode.SCATTER and f.target is not None and
                    np.hypot(*(f.xy - f.target)) < cfg.target_tolerance
                    for f in followers):
                self._flags.add("scatter_done")
                sim.log_status("SCATTER COMPLETE")

        elif st == MissionState.REGROUPING:
            if followers and all(f.mode == AgentMode.REGROUP for f in followers) \
                    and regroup_complete(sim.states(), sim.formation,
                                         cfg.regroup_tolerance):
                sim.log_status("REGROUP COMPLETE")
                self.state = MissionState.FOLLOWING
                for f in followers:                 # hand over to FOLLOW mode
                    sim.send_command(f.id, FOLLOW, self._offset_payload(sim, f.id))

        elif st == MissionState.TASK_EXECUTION:
            if sim.tasks.has_pending():
                self._do_allocate(sim, announce=False)   # re-use freed followers
            elif sim.tasks.all_done():
                sim.log_status("ALL TASKS COMPLETE")
                self._begin_regroup(sim)
            elif not sim.tasks.active():
                self.state = MissionState.FOLLOWING

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _record(self, sim, res: CommandResult) -> None:
        self.history.appendleft({
            "t": round(sim.t, 2),
            "clock": datetime.now().strftime("%H:%M:%S"),
            "text": res.command, "ok": res.ok, "message": res.message,
        })
        sim.log_status(res.message)

    @staticmethod
    def _offset_payload(sim, agent_id: int) -> dict:
        o = sim.formation[agent_id]
        return {"ox": float(o[0]), "oy": float(o[1])}

    def override_stop(self, sim) -> None:
        """Public wrapper: a new motion input cancels STOP."""
        self._override_stop(sim)

    def _override_stop(self, sim) -> None:
        """A new motion command overrides STOP (followers get RESUME first)."""
        if self.state == MissionState.STOPPED:
            sim.clear_stop()
            for f in sim.followers:
                sim.send_command(f.id, COMMAND, {"kind": "RESUME"})
            self.state = self._before_stop

    def _single_target(self, cmd: Command, sim):
        """Resolve a single-agent selector to a follower (or None)."""
        if cmd.agent in (None, 0):
            return None
        if not 1 <= cmd.agent < len(sim.agents):
            raise CommandError(f"NO SUCH AGENT: {cmd.agent}")
        return sim.agents[cmd.agent]

    def _begin_regroup(self, sim) -> None:
        sim.tasks.cancel_active()
        for f in sim.followers:
            sim.send_command(f.id, REGROUP, self._offset_payload(sim, f.id))
        self.state = MissionState.REGROUPING
        self._flags.discard("scatter_done")

    # ---- allocation shared by ALLOCATE and the auto re-allocation -------------
    def _do_allocate(self, sim, announce: bool = True) -> CommandResult:
        busy = sim.tasks.busy_agents()
        avail = {f.id: f.xy.copy() for f in sim.followers
                 if f.id not in busy and f.mode != AgentMode.STOPPED}
        if not sim.tasks.pending():
            return CommandResult(False, "NO PENDING TASKS")
        if not avail:
            return CommandResult(False, "NO AVAILABLE AGENTS")
        assignments = sim.tasks.allocate(avail)
        lines = []
        for task, aid in assignments:
            sim.send_command(aid, TASK_ASSIGN,
                             {"task_id": task.id, "x": task.x, "y": task.y,
                              "priority": task.priority})
            lines.append(f"Task {task.id} ({task.x:.0f},{task.y:.0f}) -> Agent {aid}")
        if assignments:
            self.state = MissionState.TASK_EXECUTION
            if not announce:
                sim.log_status("TASKS ALLOCATED: " + "; ".join(lines))
        return CommandResult(bool(assignments),
                             "TASK ALLOCATION: " + "; ".join(lines) if lines
                             else "NOTHING ALLOCATED")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def _cmd_goto(self, cmd: Command, sim) -> CommandResult:
        ok, xy, note = validate_target(cmd.args[0], cmd.args[1], self.cfg)
        if not ok:
            return CommandResult(False, note)
        follower = self._single_target(cmd, sim)
        if follower is not None:                    # single-agent GOTO
            sim.send_command(follower.id, COMMAND,
                             {"kind": "GOTO", "apply": True, "x": xy[0], "y": xy[1]})
            return CommandResult(True, f"{follower.name}: GOTO {xy[0]:.1f} {xy[1]:.1f}"
                                 + (f" | {note}" if note else ""))
        self._override_stop(sim)
        if self.state != MissionState.GOTO:
            self._before_goto = (self.state if self.state in (
                MissionState.SCATTERING, MissionState.REGROUPING,
                MissionState.TASK_EXECUTION) else MissionState.FOLLOWING)
        sim.leader_goto(np.array(xy))
        self.state = MissionState.GOTO
        for f in sim.followers:                     # informational broadcast
            sim.send_command(f.id, COMMAND,
                             {"kind": "GOTO", "apply": False, "x": xy[0], "y": xy[1]},
                             reliable=False)
        return CommandResult(True, f"GOTO {xy[0]:.1f} {xy[1]:.1f}"
                             + (f" | {note}" if note else ""))

    def _cmd_move(self, cmd: Command, sim) -> CommandResult:
        dx, dy = _num(cmd.args[0]), _num(cmd.args[1])
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return CommandResult(False, "MOVE REJECTED: offsets must be finite")
        base = sim.leader.xy if self._single_target(cmd, sim) is None \
            else sim.agents[cmd.agent].xy
        sub = Command("GOTO", [str(base[0] + dx), str(base[1] + dy)],
                      cmd.agent, cmd.raw)
        return self._cmd_goto(sub, sim)

    def _cmd_scatter(self, cmd: Command, sim) -> CommandResult:
        self._override_stop(sim)
        sim.tasks.cancel_active()
        tg = scatter_targets(sim.leader.state, len(sim.followers), self.cfg)
        self.scatter_targets = {}
        for f, p in zip(sim.followers, tg):
            self.scatter_targets[f.id] = p
            sim.send_command(f.id, SCATTER, {"x": float(p[0]), "y": float(p[1])})
        self.state = MissionState.SCATTERING
        self._flags.discard("scatter_done")
        return CommandResult(True, "SCATTER: dispersing followers")

    def _cmd_regroup(self, cmd: Command, sim) -> CommandResult:
        self._override_stop(sim)
        self._begin_regroup(sim)
        return CommandResult(True, "REGROUP: returning to formation")

    def _cmd_follow(self, cmd: Command, sim) -> CommandResult:
        self._override_stop(sim)
        sim.tasks.cancel_active()
        for f in sim.followers:
            sim.send_command(f.id, FOLLOW, self._offset_payload(sim, f.id))
        self._before_goto = MissionState.FOLLOWING
        self.state = (MissionState.GOTO if sim.leader.mode == AgentMode.GOTO
                      else MissionState.FOLLOWING)
        self._flags.discard("scatter_done")
        return CommandResult(True, "FOLLOW: leader-follower mode restored")

    def _cmd_stop(self, cmd: Command, sim) -> CommandResult:
        follower = self._single_target(cmd, sim)
        if follower is not None:
            sim.send_command(follower.id, STOP, {})
            return CommandResult(True, f"{follower.name}: STOP")
        if self.state == MissionState.STOPPED:
            return CommandResult(False, "ALREADY STOPPED")
        self._before_stop = self.state
        sim.stop_leader()
        for f in sim.followers:
            sim.send_command(f.id, STOP, {})
        self.state = MissionState.STOPPED
        return CommandResult(True, "STOP: all agents stopping")

    def _cmd_resume(self, cmd: Command, sim) -> CommandResult:
        follower = self._single_target(cmd, sim)
        if follower is not None:
            sim.send_command(follower.id, COMMAND, {"kind": "RESUME"})
            return CommandResult(True, f"{follower.name}: RESUME")
        if self.state != MissionState.STOPPED:
            return CommandResult(False, "NOTHING TO RESUME")
        sim.resume_leader()
        for f in sim.followers:
            sim.send_command(f.id, COMMAND, {"kind": "RESUME"})
        self.state = self._before_stop
        return CommandResult(True, f"RESUME: continuing {self.state.value}")

    def _cmd_formation(self, cmd: Command, sim) -> CommandResult:
        if not cmd.args:
            rows = "; ".join(f"A{i}=({o[0]:.2f},{o[1]:.2f})"
                             for i, o in enumerate(sim.formation) if i > 0)
            return CommandResult(True, f"FORMATION {sim.formation_name}: {rows}")
        name = cmd.args[0].upper()
        if name not in FORMATION_PRESETS:
            return CommandResult(False, f"UNKNOWN FORMATION '{cmd.args[0]}' "
                                        f"(choose {', '.join(FORMATION_PRESETS)})")
        self._override_stop(sim)
        sim.set_formation(name)
        self._begin_regroup(sim)
        return CommandResult(True, f"FORMATION {name}: regrouping")

    def _cmd_status(self, cmd: Command, sim) -> CommandResult:
        return CommandResult(True, sim.status_text())

    def _cmd_task(self, cmd: Command, sim) -> CommandResult:
        a = cmd.args
        prio = a[3] if len(a) > 3 else 1
        ok, msg = sim.tasks.add_task(a[0], _num(a[1]), _num(a[2]), prio, sim.t)
        return CommandResult(ok, msg)

    def _cmd_allocate(self, cmd: Command, sim) -> CommandResult:
        if self.state == MissionState.STOPPED:
            return CommandResult(False, "SYSTEM STOPPED: RESUME or FOLLOW first")
        return self._do_allocate(sim, announce=True)

    def _cmd_tasks(self, cmd: Command, sim) -> CommandResult:
        lines = sim.tasks.summary_lines()
        return CommandResult(True, "TASKS: " + (" | ".join(lines) if lines else "none"))

    def _cmd_cleartasks(self, cmd: Command, sim) -> CommandResult:
        busy = sim.tasks.busy_agents()
        sim.tasks.cancel_active()
        sim.tasks.clear()
        for aid in busy:
            sim.send_command(aid, REGROUP, self._offset_payload(sim, aid))
        if self.state == MissionState.TASK_EXECUTION:
            self.state = MissionState.FOLLOWING
        return CommandResult(True, "TASKS CLEARED")

    def _cmd_mode(self, cmd: Command, sim) -> CommandResult:
        try:
            mode = SimMode(cmd.args[0].upper())
        except ValueError:
            return CommandResult(False, "MODE must be AUTO, MANUAL or COMMAND")
        sim.set_sim_mode(mode)
        return CommandResult(True, f"MODE {mode.value}")
