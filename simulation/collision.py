"""
collision.py -- safety layer: inter-agent avoidance and world-boundary safety.

Boundary safety is done at three levels (spec section 11, level 1):

  1. Target validation / projection   -> nothing outside the world is ever
                                         commanded                (validate_target, clamp_target)
  2. Soft boundary-avoidance steering -> physically meaningful: robots turn
                                         away from walls and slow down
                                                                  (boundary_force, boundary_guard_command)
  3. Hard clamp                       -> last-resort numerical guard only
                                                                  (clamp_state_to_world)

The visualisation never needs to 'zoom out' to hide an escaped robot because
the simulated robot cannot escape.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .config import Config
from .kinematics import wrap_angle


# ---------------------------------------------------------------------------
# Inter-agent avoidance (artificial potential field, notebook logic)
# ---------------------------------------------------------------------------
def avoidance_vector(i: int, poses: np.ndarray, cfg: Config) -> np.ndarray:
    """Repulsive force on agent i from every other agent closer than
    `safe_distance`.  Same formula as the notebook; the magnitude is capped
    (non-legacy) so 1/d^3 growth can never explode the command."""
    force = np.zeros(2)
    for j in range(len(poses)):
        if i == j:
            continue
        dv = poses[i, :2] - poses[j, :2]
        d = np.linalg.norm(dv) + 1e-9
        if d < cfg.safe_distance:
            strength = cfg.k_avoid * (1 / d - 1 / cfg.safe_distance) / (d * d)
            force += strength * (dv / d)
            if cfg.avoid_swirl:
                # tangential component: head-on agents sidestep to the same
                # side instead of pushing straight against each other
                force += strength * cfg.avoid_swirl * np.array([-dv[1], dv[0]]) / d
    if not cfg.legacy_compat and cfg.max_force:
        mag = np.linalg.norm(force)
        if mag > cfg.max_force:
            force *= cfg.max_force / mag
    return force


def yield_speed_factor(i: int, poses: np.ndarray, cfg: Config) -> float:
    """Speed multiplier in [0, 1]: slow down (to a stop at
    `yield_min_distance`) when another agent is close AHEAD of agent i.
    Agents behind or beside do not slow it, so a blocked robot never traps
    the one it is waiting for."""
    factor = 1.0
    th = poses[i, 2]
    for j in range(len(poses)):
        if j == i:
            continue
        dx, dy = poses[j, 0] - poses[i, 0], poses[j, 1] - poses[i, 1]
        d = math.hypot(dx, dy)
        # only agents inside a ~60 degree forward cone block motion
        if d < cfg.safe_distance and (dx * math.cos(th) + dy * math.sin(th)) > 0.5 * d:
            span = max(cfg.safe_distance - cfg.yield_min_distance, 1e-6)
            factor = min(factor, float(np.clip((d - cfg.yield_min_distance) / span, 0.0, 1.0)))
    return factor


def min_separation(poses: np.ndarray) -> float:
    """Smallest pairwise distance between agents."""
    n = len(poses)
    best = np.inf
    for a in range(n):
        for b in range(a + 1, n):
            best = min(best, float(np.hypot(*(poses[a, :2] - poses[b, :2]))))
    return best


# ---------------------------------------------------------------------------
# Boundary handling
# ---------------------------------------------------------------------------
def inside_world(xy, cfg: Config, margin: float = 0.0) -> bool:
    return (cfg.world_x_min + margin <= xy[0] <= cfg.world_x_max - margin and
            cfg.world_y_min + margin <= xy[1] <= cfg.world_y_max - margin)


def boundary_force(xy, cfg: Config) -> np.ndarray:
    """Inward-pointing vector that grows from 0 (at `boundary_margin` from a
    wall) to 1 (on the wall).  Quadratic ramp, summed over the four walls."""
    m = cfg.boundary_margin
    f = np.zeros(2)
    for axis, lo, hi in ((0, cfg.world_x_min, cfg.world_x_max),
                         (1, cfg.world_y_min, cfg.world_y_max)):
        d_lo = xy[axis] - lo
        d_hi = hi - xy[axis]
        if d_lo < m:
            f[axis] += (1.0 - max(d_lo, 0.0) / m) ** 2
        if d_hi < m:
            f[axis] -= (1.0 - max(d_hi, 0.0) / m) ** 2
    return f


def boundary_guard_command(state, v_cmd: float, w_cmd: float,
                           cfg: Config, target=None) -> Tuple[float, float]:
    """Steer a (v, omega) command away from nearby walls.

    Inside the margin zone the command is blended toward 'turn to face
    inward' and the speed is reduced when the robot is moving outward.  A
    robot at the wall heading outward stops and turns in place.  Outside the
    zone the command is returned untouched.
    """
    if not cfg.boundary_guard:
        return v_cmd, w_cmd
    f = boundary_force(state[:2], cfg)
    mag = float(np.linalg.norm(f))
    if mag < 1e-9:
        return v_cmd, w_cmd
    weight = min(1.0, mag)
    if target is not None:
        # The aim point is always inside the valid world.  Fade the guard out
        # as the robot closes in on it, otherwise a target near a wall would
        # be fought by the guard and the robot would stall short of it.
        weight *= float(np.clip(np.hypot(target[0] - state[0], target[1] - state[1])
                                / cfg.boundary_margin, 0.0, 1.0))
    psi = math.atan2(f[1], f[0])                      # direction pointing inward
    moving_forward = v_cmd >= 0.0
    heading_of_motion = state[2] + (0.0 if moving_forward else math.pi)
    he = wrap_angle(psi - heading_of_motion)
    w_corr = float(np.clip(cfg.k_boundary * he, -cfg.max_w, cfg.max_w))
    new_w = (1.0 - weight) * w_cmd + weight * w_corr
    new_v = v_cmd * (1.0 - weight * (1.0 - max(0.0, math.cos(he))))
    return new_v, new_w


def clamp_target(xy, cfg: Config, margin: float | None = None) -> np.ndarray:
    """Project a world point into the valid region (inside by `margin`)."""
    if not cfg.clamp_targets:
        return np.asarray(xy, float)
    m = cfg.target_margin if margin is None else margin
    return np.array([
        float(np.clip(xy[0], cfg.world_x_min + m, cfg.world_x_max - m)),
        float(np.clip(xy[1], cfg.world_y_min + m, cfg.world_y_max - m)),
    ])


def validate_target(x, y, cfg: Config):
    """Validate an operator-supplied coordinate.

    Returns (ok, (x, y), note):
      * NaN / inf / non-numeric          -> rejected
      * outside the world                -> clamped inside, with a note
      * valid                            -> unchanged
    """
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return False, None, "TARGET REJECTED: coordinates are not numbers"
    if not (math.isfinite(x) and math.isfinite(y)):
        return False, None, "TARGET REJECTED: coordinates must be finite (no NaN/inf)"
    if inside_world((x, y), cfg, cfg.target_margin):
        return True, (x, y), ""
    cx, cy = clamp_target((x, y), cfg, max(cfg.target_margin, 0.0))
    note = (f"TARGET OUTSIDE WORLD: clamped ({x:.2f},{y:.2f}) -> "
            f"({cx:.2f},{cy:.2f})")
    return True, (float(cx), float(cy)), note


def clamp_state_to_world(state: np.ndarray, cfg: Config):
    """Final safety net: hard-limit position to the world rectangle.

    Returns (state, was_clamped).  If the clamp triggers, forward speed is
    zeroed so the robot does not keep pushing into the wall.  With the soft
    guard active this should (almost) never fire; the simulation counts it.
    """
    if not cfg.clamp_to_world:
        return state, False
    e = cfg.clamp_epsilon
    x = float(np.clip(state[0], cfg.world_x_min + e, cfg.world_x_max - e))
    y = float(np.clip(state[1], cfg.world_y_min + e, cfg.world_y_max - e))
    if x != state[0] or y != state[1]:
        out = state.copy()
        out[0], out[1] = x, y
        out[3] = 0.0
        return out, True
    return state, False
