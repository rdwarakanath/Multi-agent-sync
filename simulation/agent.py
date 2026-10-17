"""
agent.py -- agent data model.

An `Agent` is a plain data container: pose/velocity state, current behaviour
mode, current target/task, and the follower's local view of the network
(last received leader packet, last command id, ...).

It contains NO control or physics logic -- that lives in controller.py and
kinematics.py -- so the same container can later wrap a Webots robot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .kinematics import body_to_world


class AgentMode(str, Enum):
    """Per-agent behaviour mode (what the agent is doing right now)."""
    IDLE = "IDLE"          # holding still, no command
    AUTO = "AUTO"          # leader: predefined trajectory
    MANUAL = "MANUAL"      # leader: keyboard-driven
    GOTO = "GOTO"          # driving to a world coordinate
    FOLLOW = "FOLLOW"      # holding formation slot relative to the leader
    SCATTER = "SCATTER"    # dispersing to an individual safe point
    REGROUP = "REGROUP"    # returning to formation slot
    TASK = "TASK"          # executing an allocated task
    STOPPED = "STOPPED"    # commanded to stop


class LinkStatus(str, Enum):
    """Health of the leader -> agent radio link."""
    SELF = "SELF"          # the leader itself
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


@dataclass
class AgentState:
    """Convenience (x, y, theta, v, omega) view of the state vector."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0

    @classmethod
    def from_array(cls, a) -> "AgentState":
        return cls(*[float(u) for u in a[:5]])

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.omega], float)


def agent_name(agent_id: int) -> str:
    return "Leader" if agent_id == 0 else f"Agent {agent_id}"


def agent_short(agent_id: int) -> str:
    return "L" if agent_id == 0 else f"F{agent_id}"


@dataclass
class Agent:
    id: int
    state: np.ndarray                       # [x, y, theta, v, omega]
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mode: AgentMode = AgentMode.IDLE
    prev_mode: AgentMode = AgentMode.IDLE
    target: Optional[np.ndarray] = None     # world xy of the active target
    task_id: Optional[str] = None
    task_done: bool = False
    link: LinkStatus = LinkStatus.ONLINE

    # --- follower-side communication state --------------------------------
    last_rx: Optional[np.ndarray] = None    # last leader packet [x,y,th,v,w]
    rx_time: float = 0.0                    # sample time of that packet
    rx_seq: int = -1                        # highest POSE_UPDATE seq accepted
    last_cmd_id: int = 0                    # highest command id applied
    saved: Optional[dict] = None            # snapshot taken on STOP
    next_heartbeat: float = 0.0

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return agent_name(self.id)

    @property
    def short(self) -> str:
        return agent_short(self.id)

    @property
    def role(self) -> str:
        return "leader" if self.id == 0 else "follower"

    @property
    def xy(self) -> np.ndarray:
        return self.state[:2]

    @property
    def x(self) -> float: return float(self.state[0])
    @property
    def y(self) -> float: return float(self.state[1])
    @property
    def theta(self) -> float: return float(self.state[2])
    @property
    def v(self) -> float: return float(self.state[3])
    @property
    def omega(self) -> float: return float(self.state[4])

    def set_mode(self, mode: AgentMode) -> None:
        self.prev_mode = self.mode
        self.mode = mode

    def as_dict(self) -> dict:
        """JSON-friendly snapshot for UIs."""
        return {
            "id": self.id,
            "name": self.name,
            "short": self.short,
            "role": self.role,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "theta": round(self.theta, 3),
            "v": round(self.v, 3), "omega": round(self.omega, 3),
            "mode": self.mode.value,
            "task": self.task_id,
            "target": None if self.target is None
            else [round(float(self.target[0]), 3), round(float(self.target[1]), 3)],
            "link": self.link.value,
        }


def make_agents(n_agents: int, formation: np.ndarray,
                rng: np.random.Generator) -> list:
    """Create the agent list.

    The leader starts at the origin; followers start near their formation
    slots with small random pose errors.  The RNG draw order (x, y, theta per
    follower) is identical to the notebook's `initialize_agents`, so seeded
    runs start from exactly the same configuration.
    """
    states = np.zeros((n_agents, 5))
    for i in range(1, n_agents):
        xy = body_to_world(formation[i], states[0])
        states[i, :3] = [xy[0] + rng.normal(0, .10),
                         xy[1] + rng.normal(0, .10),
                         rng.normal(0, .04)]
    agents = []
    for i in range(n_agents):
        a = Agent(id=i, state=states[i].copy(), offset=np.array(formation[i], float))
        if i == 0:
            a.mode = AgentMode.AUTO
            a.link = LinkStatus.SELF
        else:
            a.mode = AgentMode.FOLLOW
            a.last_rx = states[0].copy()    # initial knowledge of the leader
            a.rx_time = 0.0
            a.next_heartbeat = 0.1 * i      # stagger heartbeats
        agents.append(a)
    return agents
