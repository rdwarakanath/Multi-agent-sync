"""
controller.py -- control laws (the 'shared control core').

Pure numeric logic: state in, (v_cmd, omega_cmd) out.  Nothing here knows
about Matplotlib, HTTP, Webots or the message bus, so it can be re-used
unchanged when the plant is swapped (HIL / Webots / firmware port).

Contents
--------
PID                    PID with conditional-integration anti-windup (unchanged)
adaptive_gains         error-magnitude gain scheduling (unchanged)
auto_leader_command    the notebook's predefined leader trajectory (AUTO mode)
go_to_point            drive-to-coordinate law (GOTO, SCATTER, TASK, follower GOTO)
FollowerController     per-follower formation tracking: dead-reckoned leader
                       slot + neighbour coupling + avoidance + adaptive PID
"""
from __future__ import annotations

import numpy as np

from .collision import clamp_target
from .config import Config
from .kinematics import (desired_heading_to_point, wrap_angle,
                         world_to_body_error)


# ---------------------------------------------------------------------------
# Building blocks carried over from the notebook
# ---------------------------------------------------------------------------
class PID:
    """PID with conditional integration (integral frozen while saturated)."""

    def __init__(self, kp, ki, kd, limit):
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.integral = 0.0
        self.prev = 0.0

    def step(self, e, dt):
        de = (e - self.prev) / dt
        candidate = self.kp * e + self.ki * self.integral + self.kd * de
        out = np.clip(candidate, -self.limit, self.limit)
        if abs(candidate) < self.limit:
            self.integral += e * dt
        self.prev = e
        return out

    def reset(self):
        self.integral = 0.0
        self.prev = 0.0


def adaptive_gains(kp, ki, kd, err):
    """Error-dependent gain scheduling: stiffer P / softer I when error is large."""
    scale = 1 + 0.75 * np.tanh(abs(err) / 0.75)
    return kp * scale, ki / (0.75 + 0.25 * scale), kd * (0.85 + 0.35 * scale)


def auto_leader_command(t):
    """Original open-loop leader trajectory (AUTO mode): returns [v_cmd, w_cmd]."""
    return np.array([0.82 + 0.16 * np.sin(0.18 * t),
                     0.28 * np.sin(0.24 * t) + 0.16 * np.sin(0.075 * t + 0.8)])


# ---------------------------------------------------------------------------
# Drive-to-point law
# ---------------------------------------------------------------------------
def go_to_point(state, target, v_max, kv, kw, max_w, tol):
    """Command that drives a unicycle to a world coordinate and stops.

    * heading controller:  omega = kw * heading_error   (saturated)
    * speed:               v = kv * distance, capped at v_max  -> smooth
                           slow-down on approach
    * cos(heading_error)+  scaling: the robot turns in place when the target
                           is behind it instead of driving away
    * arrival:             inside `tol` the command is exactly (0, 0)

    Returns (v_cmd, w_cmd, distance, arrived).  The caller still passes the
    command through the plant (RK4 + actuator lag): no teleporting.
    """
    dx, dy = target[0] - state[0], target[1] - state[1]
    dist = float(np.hypot(dx, dy))
    if dist < tol:
        return 0.0, 0.0, dist, True
    he = wrap_angle(np.arctan2(dy, dx) - state[2])
    w = float(np.clip(kw * he, -max_w, max_w))
    v = float(min(v_max, kv * dist) * max(0.0, np.cos(he)))
    return v, w, dist, False


def steer_around(pos, target, avoid, cfg: Config):
    """Aim point for go-to motion that bends the HEADING away from nearby agents.

    The avoidance force alone only shifts the aim point by a small amount,
    which is negligible for a distant target (a blocked robot would never
    turn).  Here the unit direction to the target is blended with the
    normalised avoidance force, and the aim point is placed along the result.
    """
    pos = np.asarray(pos, float)
    to_t = np.asarray(target, float) - pos
    dist = float(np.hypot(*to_t))
    mag = float(np.hypot(*avoid))
    if dist < 1e-9 or mag < 1e-9 or cfg.legacy_compat:
        return np.asarray(target, float)
    d = to_t / dist + cfg.avoid_heading_gain * np.asarray(avoid, float) / max(cfg.max_force, 1e-9)
    n = float(np.hypot(*d))
    if n < 1e-9:
        return np.asarray(target, float)
    return pos + d / n * max(dist, 1.0)


# ---------------------------------------------------------------------------
# Follower controller
# ---------------------------------------------------------------------------
class FollowerController:
    """Controller instance for ONE follower (owns its two PID loops)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pv = PID(cfg.kp, cfg.ki, cfg.kd, cfg.max_v)
        self.pw = PID(cfg.kp * .9, cfg.ki * .5, cfg.kd * .8, cfg.max_w)

    def reset(self):
        self.pv.reset()
        self.pw.reset()

    # ------------------------------------------------------------------
    def formation_command(self, state, target, neighbor_force, avoid,
                          leader_est, leader_xy_true, advanced, rng, sigma, dt):
        """FOLLOW / REGROUP law -- the notebook's follower controller.

        target         formation slot (world xy) from the estimated leader pose
        neighbor_force distributed neighbour-coupling term (advanced mode)
        avoid          collision-avoidance vector (advanced mode)
        leader_est     dead-reckoned leader state [x,y,th,v,w]
        leader_xy_true true leader position (only for the near-leader slow-down)

        Returns (v_cmd, w_cmd, effective_target, error_distance).
        """
        cfg = self.cfg
        eff = target + neighbor_force + cfg.avoid_gain * avoid
        if cfg.clamp_targets:                       # never aim outside the world
            eff = clamp_target(eff, cfg)

        e = world_to_body_error(eff, state)
        dist = float(np.linalg.norm(e))             # error BEFORE sensor noise
        e = e + rng.normal(0, sigma, 2)             # simulated sensing noise

        if advanced:
            kp, ki, kd = adaptive_gains(cfg.kp, cfg.ki, cfg.kd, dist)
        else:
            kp, ki, kd = cfg.kp, cfg.ki, cfg.kd
        self.pv.kp, self.pv.ki, self.pv.kd = kp, ki, kd
        self.pw.kp, self.pw.ki, self.pw.kd = kp * .9, ki * .5, kd * .8

        # Forward feed-forward speed.  Legacy: constant 0.72 (notebook).
        # Otherwise: the leader's actual speed, so a stationary / slow leader
        # does not make followers overshoot their slots.
        if cfg.legacy_compat:
            v_ff = cfg.v_feedforward_legacy
        else:
            v_ff = float(np.clip(leader_est[3], 0.0, cfg.max_v))
        vcmd = float(np.clip(v_ff + self.pv.step(e[0], dt), -.25, cfg.max_v))

        he = wrap_angle(desired_heading_to_point(eff, state) - state[2])
        if not cfg.legacy_compat:
            # Close to the slot the bearing-to-target is ill-defined; blend
            # toward the leader's heading to avoid jitter while holding station.
            wgt = float(np.clip(1.0 - dist / 0.6, 0.0, 1.0))
            he = (1 - wgt) * he + wgt * wrap_angle(leader_est[2] - state[2])
        wcmd = float(np.clip(
            self.pw.step(he + .55 * np.arctan2(e[1], max(abs(e[0]), .15)), dt),
            -cfg.max_w, cfg.max_w))

        if np.linalg.norm(state[:2] - leader_xy_true[:2]) < cfg.near_leader_distance:
            vcmd = min(vcmd, cfg.near_leader_speed)
        return vcmd, wcmd, eff, dist

    # ------------------------------------------------------------------
    def static_command(self, state, target, avoid, advanced):
        """SCATTER / TASK / follower-GOTO law: drive to a fixed world point.

        Collision avoidance nudges the aim point (as in formation mode);
        arrival is judged against the real target.  Returns
        (v_cmd, w_cmd, effective_target, distance, arrived).
        """
        cfg = self.cfg
        target = np.asarray(target, float)
        dist_real = float(np.hypot(*(target - state[:2])))
        if advanced:
            eff = steer_around(state[:2], target, avoid, cfg)
        else:
            eff = target
        if cfg.clamp_targets:
            eff = clamp_target(eff, cfg)
        if dist_real < cfg.target_tolerance:
            return 0.0, 0.0, eff, dist_real, True
        v, w, _d, _ = go_to_point(state, eff, cfg.follower_goto_max_v,
                                  cfg.goto_kv, cfg.goto_kw, cfg.max_w, 0.0)
        return v, w, eff, dist_real, False
