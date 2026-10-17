"""
network.py -- wireless network model (ESP-NOW stand-in).

Contents
--------
GilbertElliottChannel   two-state burst-loss channel + random latency
                        (logic unchanged from the notebook)
dead_reckon             leader-state extrapolation during blackouts
                        (notebook logic + optional exponential decay, Eq. 26)
LinkStats               per-link counters and recent-loss window
Network                 one directed channel per (sender, receiver) pair,
                        in-flight delivery queue, statistics, link health

The Network is the single place where packets are lost or delayed.  Every
decision is reported back as a `CommEvent` and stored on the MessageBus, so
the UI shows exactly what the model did.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import numpy as np

from .agent import LinkStatus
from .config import Config
from .kinematics import wrap_angle
from .messages import (CommEvent, Message, MessageBus, STATUS_DELAYED,
                       STATUS_LOST, STATUS_OK)


# ---------------------------------------------------------------------------
# Channel model (unchanged maths)
# ---------------------------------------------------------------------------
class GilbertElliottChannel:
    """Two-state (good/bad) burst-loss channel with random latency.

    `transmit(packet)` returns None when the packet is lost, otherwise
    (packet, delay_in_steps).  RNG usage order is identical to the notebook.
    """

    def __init__(self, p_gb=0.025, p_bg=0.22, seed=1,
                 base_delay_steps=1, max_delay_steps=4):
        self.rng = np.random.default_rng(seed)
        self.bad = False
        self.p_gb = min(max(p_gb, 0.0), 1.0)
        self.p_bg = p_bg
        self.base_delay_steps = base_delay_steps
        self.max_delay_steps = max_delay_steps
        self.force_loss = False       # test hook: behave as 100 % loss

    def transmit(self, packet):
        if self.bad:
            if self.rng.random() < self.p_bg:
                self.bad = False
        elif self.rng.random() < self.p_gb:
            self.bad = True
        if self.bad or self.force_loss:
            return None
        delay = int(self.rng.integers(self.base_delay_steps,
                                      self.max_delay_steps + 1))
        out = packet.copy() if hasattr(packet, "copy") else packet
        return out, delay


def dead_reckon(last_packet, age, use_pidr=True, lam=0.0):
    """Extrapolate the leader's last known state forward by `age` seconds.

    use_pidr=False  -> zero-order hold (ZOH): just return the stale packet.
    use_pidr=True   -> propagate through the unicycle kinematics.
    lam > 0         -> velocities decay as exp(-lam * t) (report Eq. 26) so a
                       long blackout converges to a stationary hold instead of
                       integrating a stale velocity forever.  Implemented with
                       the closed-form 'effective age' (1 - e^{-lam a}) / lam,
                       which is exact because v and omega decay together.
    lam = 0         -> original notebook behaviour.
    """
    p = np.array(last_packet, dtype=float).copy()
    if not use_pidr:
        return p
    if lam > 0.0:
        age = (1.0 - np.exp(-lam * age)) / lam
    x, y, th, v, w = p
    if abs(w) < 1e-7:
        x += v * np.cos(th) * age
        y += v * np.sin(th) * age
    else:
        x += (v / w) * (np.sin(th + w * age) - np.sin(th))
        y += (v / w) * (-np.cos(th + w * age) + np.cos(th))
        th += w * age
    p[:3] = [x, y, wrap_angle(th)]
    return p


# ---------------------------------------------------------------------------
# Per-link statistics
# ---------------------------------------------------------------------------
@dataclass
class LinkStats:
    window: int = 40
    sent: int = 0
    received: int = 0
    lost: int = 0
    latency_sum: float = 0.0
    latency_max: float = 0.0
    last_rx_time: float = 0.0
    recent_lost: Deque[bool] = field(default_factory=deque)

    def __post_init__(self):
        self.recent_lost = deque(maxlen=self.window)

    @property
    def loss_rate(self) -> float:
        return self.lost / self.sent if self.sent else 0.0

    @property
    def avg_latency(self) -> float:
        return self.latency_sum / self.received if self.received else 0.0

    @property
    def recent_loss_rate(self) -> float:
        return (sum(self.recent_lost) / len(self.recent_lost)
                if self.recent_lost else 0.0)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class Network:
    """All radio links of the swarm.

    Links are created lazily.  Leader -> follower channels are seeded exactly
    like the notebook (seed + 100 + follower_id); the reverse channels
    (follower -> leader: ACK, HEARTBEAT) use seed + 200 + follower_id.
    """

    def __init__(self, cfg: Config, bus: MessageBus, seed: int = 42,
                 loss_scale: float = 1.0):
        self.cfg = cfg
        self.bus = bus
        self.seed = seed
        self.loss_scale = loss_scale
        self._channels: Dict[Tuple[int, int], GilbertElliottChannel] = {}
        self._stats: Dict[Tuple[int, int], LinkStats] = {}
        self._queue: list = []          # heap of (deliver_time, counter, msg)
        self._counter = 0
        self.forced_loss_all = False

    # ------------------------------------------------------------------
    def _channel(self, link: Tuple[int, int]) -> GilbertElliottChannel:
        if link not in self._channels:
            s, r = link
            follower = r if s == 0 else s
            offset = 100 if s == 0 else 200
            ch = GilbertElliottChannel(
                self.cfg.packet_good_to_bad * self.loss_scale,
                self.cfg.packet_bad_to_good,
                self.seed + offset + follower,
                self.cfg.base_delay_steps, self.cfg.max_delay_steps)
            ch.force_loss = self.forced_loss_all
            self._channels[link] = ch
        return self._channels[link]

    def _link_stats(self, link: Tuple[int, int]) -> LinkStats:
        if link not in self._stats:
            self._stats[link] = LinkStats(window=self.cfg.link_window)
        return self._stats[link]

    # ------------------------------------------------------------------
    def transmit(self, msg: Message, now: float, dt: float | None = None) -> CommEvent:
        """Send `msg` through its link; returns (and logs) what happened."""
        dt = self.cfg.dt if dt is None else dt
        link = (msg.sender, msg.receiver)
        stats = self._link_stats(link)
        stats.sent += 1

        result = self._channel(link).transmit(msg)
        if result is None:                                   # ---- LOST
            stats.lost += 1
            stats.recent_lost.append(True)
            ev = CommEvent(now, msg.sender, msg.receiver, msg.msg_type,
                           STATUS_LOST, 0.0, msg.sequence)
            self.bus.record(ev)
            return ev

        _, delay_steps = result                              # ---- DELIVERED
        latency = delay_steps * dt
        stats.received += 1
        stats.recent_lost.append(False)
        stats.latency_sum += latency
        stats.latency_max = max(stats.latency_max, latency)
        stats.last_rx_time = now + (latency if self.cfg.causal_delivery else 0.0)

        if self.cfg.causal_delivery:
            deliver_at = now + latency      # packet really arrives later
        else:
            # Legacy notebook semantics: the packet is usable immediately but
            # its data is considered `latency` seconds old.
            deliver_at = now
            msg.timestamp = now - latency

        self._counter += 1
        heapq.heappush(self._queue, (deliver_at, self._counter, msg))
        status = (STATUS_DELAYED if latency >= self.cfg.delayed_threshold_s
                  else STATUS_OK)
        ev = CommEvent(now, msg.sender, msg.receiver, msg.msg_type,
                       status, latency, msg.sequence)
        self.bus.record(ev)
        return ev

    def poll(self, now: float) -> List[Message]:
        """Return all in-flight messages whose delivery time has arrived."""
        out = []
        while self._queue and self._queue[0][0] <= now + 1e-9:
            out.append(heapq.heappop(self._queue)[2])
        return out

    # ------------------------------------------------------------------
    # Test / experiment hooks
    # ------------------------------------------------------------------
    def set_forced_loss(self, enabled: bool, link: Tuple[int, int] | None = None):
        """Force 100 % packet loss on one link, or on every link."""
        if link is None:
            self.forced_loss_all = enabled
            for ch in self._channels.values():
                ch.force_loss = enabled
        else:
            self._channel(link).force_loss = enabled

    # ------------------------------------------------------------------
    # Health + statistics (values come from the real model)
    # ------------------------------------------------------------------
    def link_status(self, receiver: int, now: float) -> LinkStatus:
        st = self._stats.get((0, receiver))
        if st is None or st.sent == 0:
            return LinkStatus.ONLINE
        silent = now - st.last_rx_time
        if silent > self.cfg.offline_timeout_s:
            return LinkStatus.OFFLINE
        if (st.recent_loss_rate > self.cfg.degraded_loss_rate
                or silent > self.cfg.degraded_age_s):
            return LinkStatus.DEGRADED
        return LinkStatus.ONLINE

    def snapshot(self) -> dict:
        """Aggregate and per-link statistics for the UI panel."""
        sent = sum(s.sent for s in self._stats.values())
        recv = sum(s.received for s in self._stats.values())
        lost = sum(s.lost for s in self._stats.values())
        lat_sum = sum(s.latency_sum for s in self._stats.values())
        lat_max = max([s.latency_max for s in self._stats.values()] or [0.0])
        links = {}
        for (s, r), st in self._stats.items():
            if s == 0:
                links[r] = {"sent": st.sent, "received": st.received,
                            "lost": st.lost,
                            "loss_rate": round(100 * st.loss_rate, 2),
                            "avg_latency": round(st.avg_latency, 4)}
        return {
            "sent": sent, "received": recv, "lost": lost,
            "loss_rate": round(100.0 * lost / sent, 2) if sent else 0.0,
            "avg_latency": round(lat_sum / recv, 4) if recv else 0.0,
            "max_latency": round(lat_max, 4),
            "links": links,
            "forced_loss": self.forced_loss_all,
        }
