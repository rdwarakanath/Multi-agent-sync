"""
logger.py -- structured CSV logging (spec section 24).

One row per agent per simulation tick.  Columns:

    timestamp, agent_id, x, y, theta, v, omega, mode, target_x, target_y,
    packet_sent, packet_received, packet_lost, packet_latency, task_id

Packet columns count the traffic of THIS tick:
    packet_sent      messages sent BY this agent
    packet_received  messages addressed TO this agent that were delivered
    packet_lost      messages addressed TO this agent that were lost
    packet_latency   mean latency [s] of those delivered messages

Logging is optional and off by default (research runs do not write files):

    sim.attach_logger(SimLogger())         # writes to logs/sim_<time>.csv
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional

from .messages import STATUS_LOST

FIELDS = ["timestamp", "agent_id", "x", "y", "theta", "v", "omega", "mode",
          "target_x", "target_y", "packet_sent", "packet_received",
          "packet_lost", "packet_latency", "task_id"]


class SimLogger:
    def __init__(self, directory: str = "logs", filename: Optional[str] = None,
                 flush_every: int = 200):
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            filename = datetime.now().strftime("sim_%Y%m%d_%H%M%S.csv")
        self.path = os.path.join(directory, filename)
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(FIELDS)
        self._rows = 0
        self._flush_every = flush_every

    # ------------------------------------------------------------------
    def log_tick(self, t: float, agents, events) -> None:
        """`events`: CommEvents produced during this tick."""
        for a in agents:
            sent = sum(1 for e in events if e.sender == a.id)
            to_me = [e for e in events if e.receiver == a.id]
            lost = sum(1 for e in to_me if e.status == STATUS_LOST)
            ok = [e for e in to_me if e.status != STATUS_LOST]
            lat = sum(e.latency for e in ok) / len(ok) if ok else 0.0
            tx = "" if a.target is None else f"{a.target[0]:.4f}"
            ty = "" if a.target is None else f"{a.target[1]:.4f}"
            self._writer.writerow([
                f"{t:.3f}", a.id, f"{a.x:.4f}", f"{a.y:.4f}", f"{a.theta:.4f}",
                f"{a.v:.4f}", f"{a.omega:.4f}", a.mode.value, tx, ty,
                sent, len(ok), lost, f"{lat:.4f}", a.task_id or ""])
            self._rows += 1
        if self._rows >= self._flush_every:
            self._fh.flush()
            self._rows = 0

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
