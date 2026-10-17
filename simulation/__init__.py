"""
simulation -- reusable multi-agent digital-twin engine.

Typical use
-----------
    from simulation import MultiAgentSimulation

    sim = MultiAgentSimulation()
    sim.submit_command("GOTO 10 5")
    for _ in range(400):
        sim.step()
    state = sim.get_state()

Research mode (notebook-compatible batch run):

    from simulation import simulate, compute_metrics
    h = simulate("advanced", seed=101)
    print(compute_metrics(h))
"""
from .agent import Agent, AgentMode, AgentState, LinkStatus
from .commands import (Command, CommandError, CommandManager, CommandResult,
                       MissionState, SimMode, parse_command)
from .config import (CFG, DT, FORMATION, LEADER, N_AGENTS, T_FINAL,
                     WORLD_LIMITS, WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX,
                     WORLD_Y_MIN, Config, follower_ids)
from .controller import PID, FollowerController, adaptive_gains, auto_leader_command, go_to_point
from .formation import make_formation, scatter_targets
from .keyboard import KeyboardController
from .logger import SimLogger
from .messages import CommEvent, Message, MessageBus
from .metrics import (compute_metrics, monte_carlo, robustness_sweep,
                      routh_hurwitz_cubic, stability_scan, summarize_monte_carlo)
from .network import GilbertElliottChannel, Network, dead_reckon
from .simulation import MultiAgentSimulation, simulate
from .tasks import Allocator, NearestAvailableAllocator, Task, TaskManager, TaskStatus

__all__ = [
    "MultiAgentSimulation", "simulate", "Config", "CFG",
    "Agent", "AgentMode", "AgentState", "LinkStatus",
    "CommandManager", "CommandResult", "MissionState", "SimMode", "parse_command",
    "TaskManager", "Task", "TaskStatus", "Allocator", "NearestAvailableAllocator",
    "Network", "GilbertElliottChannel", "dead_reckon",
    "Message", "MessageBus", "CommEvent",
    "KeyboardController", "SimLogger",
    "compute_metrics", "monte_carlo", "robustness_sweep", "stability_scan",
    "routh_hurwitz_cubic", "summarize_monte_carlo",
    "make_formation", "scatter_targets", "FollowerController", "PID",
]
