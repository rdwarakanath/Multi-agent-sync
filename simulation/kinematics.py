"""
kinematics.py -- plant model and geometry helpers.

Agent state vector:  s = [x, y, theta, v, omega]

Plant: nonholonomic unicycle with first-order actuator lag
    x'     = v cos(theta)
    y'     = v sin(theta)
    theta' = omega
    v'     = (v_cmd - v)     / tau
    omega' = (omega_cmd - omega) / tau
integrated with classical RK4 (unchanged from the original notebook).

This module is the ONLY place that knows how a command becomes motion.  When
the project migrates to Webots, `rk4_step` is the function that gets replaced
by the Webots physics step; everything above it stays the same.
"""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def rk4_step(state, command, dt, tau):
    """One RK4 step of the unicycle + actuator-lag plant.

    `command` = (v_cmd, omega_cmd).  Returns the new 5-element state.
    """
    vc, wc = command

    def f(s):
        _x, _y, th, v, w = s
        return np.array([v * np.cos(th), v * np.sin(th), w,
                         (vc - v) / tau, (wc - w) / tau])

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def body_to_world(offset, pose):
    """Map a body-frame offset (ox, oy) of a robot with `pose` to world xy.

    Used for formation slots: leader pose + offset rotated by leader heading.
    """
    x, y, th = pose[:3]
    c, s = np.cos(th), np.sin(th)
    return np.array([x + c * offset[0] - s * offset[1],
                     y + s * offset[0] + c * offset[1]])


def world_to_body_error(target_xy, pose):
    """Position error (world target - robot) expressed in the robot body frame.

    Returns [e_x (along heading), e_y (lateral)]  -- Eq. 12 of the report.
    """
    dx, dy = target_xy[0] - pose[0], target_xy[1] - pose[1]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy])


def desired_heading_to_point(target_xy, pose):
    """World-frame bearing from the robot to `target_xy`."""
    return np.arctan2(target_xy[1] - pose[1], target_xy[0] - pose[0])


def distance(a, b) -> float:
    """Euclidean distance between two xy points (or poses)."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def saturate_command(v_cmd, w_cmd, max_v, max_w, min_v=None):
    """Clip a (v, omega) command to actuator limits."""
    lo = -max_v if min_v is None else min_v
    return (float(np.clip(v_cmd, lo, max_v)),
            float(np.clip(w_cmd, -max_w, max_w)))


def is_valid_state(state, max_abs: float = 1.0e4) -> bool:
    """False if the state contains NaN/Inf or absurdly large coordinates.

    The simulation calls this after every integration step so that a
    numerical blow-up is detected and reported instead of corrupting the
    animation (spec: 'invalid state check').
    """
    s = np.asarray(state, dtype=float)
    if not np.all(np.isfinite(s)):
        return False
    return bool(abs(s[0]) <= max_abs and abs(s[1]) <= max_abs)
