"""
tasks.py -- task model and task allocation.

The allocation strategy is a pluggable `Allocator`.  The default
`NearestAvailableAllocator` is deliberately simple; swap in a Hungarian or
auction allocator later by subclassing `Allocator` -- nothing else changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .collision import validate_target
from .config import Config


class TaskStatus(str, Enum):
    PENDING = "PENDING"          # created, not yet assigned
    ASSIGNED = "ASSIGNED"        # assignment sent to a follower
    IN_PROGRESS = "IN_PROGRESS"  # follower reports it is working on it
    DONE = "DONE"                # follower reported arrival
    CANCELLED = "CANCELLED"


@dataclass
class Task:
    id: str
    x: float
    y: float
    priority: int = 1            # 1 = most urgent
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[int] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def as_dict(self) -> dict:
        return {"id": self.id, "x": round(self.x, 2), "y": round(self.y, 2),
                "priority": self.priority, "status": self.status.value,
                "agent": self.assigned_to}


# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------
class Allocator(ABC):
    """Strategy interface: decide which available agent gets which task."""

    @abstractmethod
    def allocate(self, tasks: List[Task],
                 agents: Dict[int, np.ndarray]) -> Dict[str, int]:
        """tasks   : PENDING tasks
        agents  : {agent_id: xy position} of AVAILABLE agents
        returns : {task_id: agent_id}  (each agent at most once)"""


class NearestAvailableAllocator(Allocator):
    """Greedy: tasks in priority order, each takes the nearest free follower
    (minimum straight-line travel distance)."""

    def allocate(self, tasks, agents):
        free = dict(agents)
        result: Dict[str, int] = {}
        for task in sorted(tasks, key=lambda t: (t.priority, t.created_at)):
            if not free:
                break
            best = min(free, key=lambda a: float(np.hypot(*(free[a] - task.xy))))
            result[task.id] = best
            del free[best]
        return result


# ---------------------------------------------------------------------------
# Task manager
# ---------------------------------------------------------------------------
_ACTIVE = (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)


class TaskManager:
    def __init__(self, cfg: Config, allocator: Allocator | None = None):
        self.cfg = cfg
        self.allocator = allocator or NearestAvailableAllocator()
        self.tasks: Dict[str, Task] = {}

    # ------------------------------------------------------------------
    def add_task(self, task_id, x, y, priority=1, now: float = 0.0
                 ) -> Tuple[bool, str]:
        task_id = str(task_id).strip()
        if not task_id:
            return False, "TASK REJECTED: empty id"
        ok, xy, note = validate_target(x, y, self.cfg)
        if not ok:
            return False, note
        try:
            priority = int(priority)
        except (TypeError, ValueError):
            return False, "TASK REJECTED: priority must be an integer"
        old = self.tasks.get(task_id)
        if old is not None and old.status != TaskStatus.PENDING:
            return False, f"TASK REJECTED: id '{task_id}' already {old.status.value}"
        self.tasks[task_id] = Task(task_id, xy[0], xy[1], priority,
                                   created_at=now)
        msg = f"TASK {task_id} added at ({xy[0]:.1f},{xy[1]:.1f}) priority {priority}"
        return True, (msg + " | " + note) if note else msg

    def allocate(self, available: Dict[int, np.ndarray]) -> List[Tuple[Task, int]]:
        """Assign PENDING tasks to AVAILABLE agents; returns [(task, agent_id)]."""
        pending = self.pending()
        if not pending or not available:
            return []
        mapping = self.allocator.allocate(pending, available)
        out = []
        for tid, aid in mapping.items():
            t = self.tasks[tid]
            t.status, t.assigned_to = TaskStatus.ASSIGNED, aid
            out.append((t, aid))
        return out

    # ---- lifecycle updates driven by follower reports ----------------------
    def mark_in_progress(self, task_id: str) -> None:
        t = self.tasks.get(task_id)
        if t and t.status == TaskStatus.ASSIGNED:
            t.status = TaskStatus.IN_PROGRESS

    def complete(self, task_id: str, now: float) -> bool:
        t = self.tasks.get(task_id)
        if t and t.status in _ACTIVE:
            t.status, t.completed_at = TaskStatus.DONE, now
            return True
        return False

    def cancel_active(self) -> None:
        for t in self.tasks.values():
            if t.status in _ACTIVE:
                t.status = TaskStatus.CANCELLED

    def clear(self) -> None:
        self.tasks.clear()

    # ---- queries -----------------------------------------------------------
    def pending(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]

    def active(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.status in _ACTIVE]

    def busy_agents(self) -> set:
        return {t.assigned_to for t in self.active()}

    def has_pending(self) -> bool:
        return bool(self.pending())

    def all_done(self) -> bool:
        live = [t for t in self.tasks.values() if t.status != TaskStatus.CANCELLED]
        return bool(live) and all(t.status == TaskStatus.DONE for t in live)

    def to_list(self) -> List[dict]:
        return [t.as_dict() for t in self.tasks.values()]

    def summary_lines(self) -> List[str]:
        """Text for the TASK ALLOCATION panel."""
        lines = []
        for t in self.tasks.values():
            who = f"Agent {t.assigned_to}" if t.assigned_to is not None else "unassigned"
            lines.append(f"Task {t.id} ({t.x:.0f},{t.y:.0f}) -> {who}  [{t.status.value}]")
        return lines
