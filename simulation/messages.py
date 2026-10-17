"""
messages.py -- message abstraction and communication-event log.

Layering (spec section 22):

    simulation  --creates-->  Message
    network     --decides-->  delivered / lost / delayed   (-> CommEvent)
    UI          --only displays-->  CommEvent records

The bus never decides anything about delivery; it only numbers messages and
remembers what the network layer reported.  Because the log is filled from the
network's real decisions, the communication panel can never show fake traffic.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Tuple

from .agent import agent_name, agent_short

# ---- message types (spec section 13) ---------------------------------------
POSE_UPDATE = "POSE_UPDATE"
COMMAND = "COMMAND"
TASK_ASSIGN = "TASK_ASSIGN"
SCATTER = "SCATTER"
REGROUP = "REGROUP"
FOLLOW = "FOLLOW"
STOP = "STOP"
ACK = "ACK"
HEARTBEAT = "HEARTBEAT"

ALL_TYPES = (POSE_UPDATE, COMMAND, TASK_ASSIGN, SCATTER, REGROUP,
             FOLLOW, STOP, ACK, HEARTBEAT)

# ---- packet status values reported by the network ---------------------------
STATUS_OK = "OK"
STATUS_DELAYED = "DELAYED"
STATUS_LOST = "LOST"


@dataclass
class Message:
    """One application-level packet (sender/receiver are agent ids)."""
    sender: int
    receiver: int
    msg_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0       # time the payload data was sampled [s]
    sequence: int = 0            # per (sender, receiver) link counter

    @property
    def sender_name(self) -> str:
        return agent_name(self.sender)

    @property
    def receiver_name(self) -> str:
        return agent_name(self.receiver)


@dataclass
class CommEvent:
    """What the network did with one message (this is what the UI shows)."""
    time: float
    sender: int
    receiver: int
    msg_type: str
    status: str                  # OK | DELAYED | LOST
    latency: float               # seconds (0.0 for lost packets)
    sequence: int

    def short(self) -> str:
        """e.g.  'L -> F2 | POSE_UPDATE | DELAY=0.15s'"""
        tail = {STATUS_OK: "OK",
                STATUS_LOST: "LOST",
                STATUS_DELAYED: f"DELAY={self.latency:.2f}s"}[self.status]
        return (f"{agent_short(self.sender)} -> {agent_short(self.receiver)} | "
                f"{self.msg_type} | {tail}")

    def as_dict(self) -> dict:
        return {
            "t": round(self.time, 3),
            "from": agent_short(self.sender),
            "to": agent_short(self.receiver),
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.msg_type,
            "status": self.status,
            "latency": round(self.latency, 3),
            "seq": self.sequence,
        }


class MessageBus:
    """Numbers messages and keeps a bounded log of network outcomes."""

    def __init__(self, log_size: int = 300):
        self._seq: Dict[Tuple[int, int], int] = {}
        self.events: Deque[CommEvent] = deque(maxlen=log_size)
        self.total_events = 0

    # ------------------------------------------------------------------
    def create(self, sender: int, receiver: int, msg_type: str,
               payload: Dict[str, Any] | None = None,
               timestamp: float = 0.0) -> Message:
        """Build a message with the next sequence number of its link."""
        key = (sender, receiver)
        seq = self._seq.get(key, 0) + 1
        self._seq[key] = seq
        return Message(sender, receiver, msg_type, dict(payload or {}),
                       timestamp, seq)

    def record(self, event: CommEvent) -> None:
        self.events.append(event)
        self.total_events += 1

    def recent(self, n: int = 40, hide_pose: bool = False) -> List[dict]:
        """Newest-last list of the most recent events (dicts for UIs)."""
        ev = [e for e in self.events
              if not (hide_pose and e.msg_type == POSE_UPDATE)]
        return [e.as_dict() for e in ev[-n:]]

    def clear(self) -> None:
        self.events.clear()
        self._seq.clear()
        self.total_events = 0
