"""
formation.py -- formation geometry, SCATTER targets and REGROUP logic.

Formation offsets are expressed in the LEADER body frame and converted to
world slots with  slot = leader_pose + R(theta_L) * offset  (`body_to_world`),
exactly as in the report (Eq. 23) and the notebook.
"""
from __future__ import annotations

import numpy as np

from .collision import clamp_target
from .config import FORMATION, Config
from .kinematics import body_to_world

FORMATION_PRESETS = ("WEDGE", "LINE", "COLUMN")


def make_formation(n_agents: int, preset: str = "WEDGE") -> np.ndarray:
    """Return an (n_agents, 2) array of body-frame offsets; row 0 = leader.

    WEDGE   default chevron.  For n=5 this is exactly the notebook FORMATION.
    LINE    followers abreast of the leader (side-by-side line).
    COLUMN  single file behind the leader (convoy).
    """
    preset = preset.upper()
    if preset not in FORMATION_PRESETS:
        raise ValueError(f"unknown formation '{preset}', choose {FORMATION_PRESETS}")
    if preset == "WEDGE" and n_agents == len(FORMATION):
        return FORMATION.copy()

    off = np.zeros((n_agents, 2))
    for k in range(1, n_agents):
        row = (k + 1) // 2
        side = 1.0 if k % 2 == 1 else -1.0
        if preset == "WEDGE":
            off[k] = [-1.25 * row, side * 0.85 * row]
        elif preset == "LINE":
            off[k] = [-0.40, side * 1.30 * row]
        else:  # COLUMN
            off[k] = [-1.40 * k, 0.0]
    return off


def formation_targets(leader_pose, formation: np.ndarray) -> np.ndarray:
    """World-frame slot for every agent given the leader pose."""
    return np.array([body_to_world(o, leader_pose) for o in formation])


def fit_shift(points: np.ndarray, cfg: Config) -> np.ndarray:
    """Rigid translation that brings a whole formation footprint inside the
    world (inside by `target_margin`).

    Clamping slots one by one near a wall squeezes them onto each other and
    causes pile-ups.  Translating the formation as a rigid body preserves the
    spacing.  If the footprint is larger than the world on an axis, no shift
    is applied on that axis (per-slot clamping remains as the fallback).
    """
    m = cfg.target_margin
    lo, hi = points.min(axis=0), points.max(axis=0)
    shift = np.zeros(2)
    for ax, (a, b) in enumerate(((cfg.world_x_min, cfg.world_x_max),
                                 (cfg.world_y_min, cfg.world_y_max))):
        if hi[ax] - lo[ax] > (b - a) - 2 * m:
            continue
        if lo[ax] < a + m:
            shift[ax] = a + m - lo[ax]
        elif hi[ax] > b - m:
            shift[ax] = b - m - hi[ax]
    return shift


def formation_errors(states: np.ndarray, formation: np.ndarray) -> np.ndarray:
    """Distance of each FOLLOWER from its slot (ground truth, not estimated)."""
    return np.array([
        np.linalg.norm(states[i, :2] - body_to_world(formation[i], states[0]))
        for i in range(1, len(states))
    ])


def formation_rms(states: np.ndarray, formation: np.ndarray) -> float:
    e = formation_errors(states, formation)
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else 0.0


def regroup_complete(states: np.ndarray, formation: np.ndarray, tol: float) -> bool:
    """True when every follower is within `tol` of its formation slot."""
    e = formation_errors(states, formation)
    return bool(len(e) and np.all(e < tol))


def scatter_targets(leader_pose, n_followers: int, cfg: Config) -> np.ndarray:
    """One safe dispersal point per follower, generated around the leader.

    * Points sit on a ring of radius `scatter_radius` around a centre.
    * The centre is the leader position, SHIFTED inward if the ring would
      cross a wall (so the pattern keeps its spacing instead of being
      squashed against the boundary).
    * Every point is finally projected into the valid world.

    For 4 followers the layout follows the spec diagram:
        F1 above, F2 left, F3 right, F4 below the centre.
    Targets are anchored in world coordinates at command time.
    """
    r = cfg.scatter_radius
    m = cfg.target_margin
    cx = float(np.clip(leader_pose[0], cfg.world_x_min + r + m, cfg.world_x_max - r - m))
    cy = float(np.clip(leader_pose[1], cfg.world_y_min + r + m, cfg.world_y_max - r - m))
    if cfg.world_x_max - cfg.world_x_min < 2 * (r + m):      # tiny world fallback
        cx = 0.5 * (cfg.world_x_min + cfg.world_x_max)
    if cfg.world_y_max - cfg.world_y_min < 2 * (r + m):
        cy = 0.5 * (cfg.world_y_min + cfg.world_y_max)

    if n_followers == 4:
        angles = np.radians([90.0, 180.0, 0.0, 270.0])
    else:
        angles = np.radians(90.0 + 360.0 * np.arange(n_followers) / max(n_followers, 1))

    pts = np.array([[cx + r * np.cos(a), cy + r * np.sin(a)] for a in angles])
    return np.array([clamp_target(p, cfg) for p in pts])
