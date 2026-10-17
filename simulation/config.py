"""
config.py -- central configuration for the Multi-Agent Digital Twin.

Everything tunable lives here: world bounds, timing, controller gains,
network model parameters, safety margins and interactive-layer settings.

The first block of `Config` fields is carried over unchanged from the original
notebook so existing behaviour (ZOH / PIDR / Advanced comparison, Monte-Carlo
study, ...) is preserved.  Fields below the "NEW" marker belong to the
interactive command/communication layer.

No other simulation module defines magic numbers; they read them from `Config`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# --------------------------------------------------------------------------
# World bounds (explicit, per spec).  Boundary safety uses these everywhere:
# target validation, boundary-avoidance force and the final clamp guard.
# --------------------------------------------------------------------------
WORLD_X_MIN = -4.0
WORLD_X_MAX = 22.0
WORLD_Y_MIN = -7.0
WORLD_Y_MAX = 9.0
WORLD_LIMITS = (WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX)

# --------------------------------------------------------------------------
# Timing / population defaults (same as the notebook)
# --------------------------------------------------------------------------
DT = 0.05            # control + integration step [s]  (20 Hz)
T_FINAL = 45.0       # length of a batch (research-mode) run [s]
SEED = 42
N_AGENTS = 5         # 1 leader + 4 followers
LEADER = 0           # agent id of the leader

# Default 5-agent wedge formation, offsets expressed in the leader body frame
FORMATION = np.array([
    [0.0, 0.0],
    [-1.25, 0.85],
    [-1.25, -0.85],
    [-2.50, 1.70],
    [-2.50, -1.70],
])


def follower_ids(n_agents: int = N_AGENTS) -> np.ndarray:
    """Agent ids of all followers (1 .. n_agents-1)."""
    return np.arange(1, n_agents)


@dataclass
class Config:
    # ---- ORIGINAL notebook parameters (unchanged defaults) ---------------
    kp: float = 1.8
    ki: float = 0.18
    kd: float = 0.65
    k_neighbor: float = 0.55
    k_avoid: float = 0.75
    safe_distance: float = 0.70
    max_v: float = 1.25
    max_w: float = 2.4
    tau_motor: float = 0.12
    sensor_sigma: float = 0.025
    packet_good_to_bad: float = 0.025
    packet_bad_to_good: float = 0.22
    base_delay_steps: int = 1
    max_delay_steps: int = 4

    # ---- NEW: world / timing --------------------------------------------
    dt: float = DT
    n_agents: int = N_AGENTS
    world_x_min: float = WORLD_X_MIN
    world_x_max: float = WORLD_X_MAX
    world_y_min: float = WORLD_Y_MIN
    world_y_max: float = WORLD_Y_MAX
    max_abs_coord: float = 1.0e4        # |x|,|y| beyond this = invalid state
    collision_distance: float = 0.38    # separation below this = collision flag

    # ---- NEW: boundary safety -------------------------------------------
    boundary_guard: bool = True         # soft boundary-avoidance steering
    boundary_margin: float = 2.0        # width of the soft repulsion zone [m]
    k_boundary: float = 2.5             # heading-correction gain in that zone
    target_margin: float = 0.6          # targets are projected this far inside
    clamp_targets: bool = True
    clamp_to_world: bool = True         # final hard clamp (numerical guard only)
    clamp_epsilon: float = 0.05         # hard clamp sits this far inside
    max_force: float = 3.0              # cap on avoidance force magnitude
    avoid_gain: float = 0.6             # avoidance -> aim-point shift (legacy: 0.25)
    avoid_heading_gain: float = 1.5     # how strongly avoidance bends the heading
    avoid_swirl: float = 0.6            # tangential share of avoidance (pass on one side)
    yield_min_distance: float = 0.45    # agents ahead closer than this block motion

    # ---- NEW: GOTO / static-target motion --------------------------------
    goto_tolerance: float = 0.25        # leader arrival radius [m]
    goto_max_v: float = 1.00            # leader speed limit during GOTO
    goto_kv: float = 0.80               # speed per metre of remaining distance
    goto_kw: float = 2.00               # heading gain
    target_tolerance: float = 0.30      # follower arrival radius (scatter/task)
    follower_goto_max_v: float = 1.10

    # ---- NEW: formation / scatter / regroup -------------------------------
    scatter_radius: float = 3.0
    regroup_tolerance: float = 0.35     # each follower within this -> complete
    near_leader_distance: float = 0.45  # follower slow-down radius (original)
    near_leader_speed: float = 0.10
    v_feedforward_legacy: float = 0.72  # original constant feed-forward speed

    # ---- NEW: manual (keyboard) control -----------------------------------
    manual_max_v: float = 1.20
    manual_max_reverse: float = 0.60
    manual_max_w: float = 1.60
    manual_accel_v: float = 1.00        # m/s^2 while W/S held
    manual_decay_v: float = 1.50        # m/s^2 coast-down when released
    manual_accel_w: float = 6.00        # rad/s^2 while A/D held
    manual_decay_w: float = 8.00

    # ---- NEW: communication layer ------------------------------------------
    causal_delivery: bool = True        # packets arrive AFTER their latency
    dr_lambda: float = 0.12             # dead-reckoning velocity decay [1/s]
    heartbeat_period: float = 0.5       # follower -> leader heartbeat [s]
    cmd_retry_period: float = 0.25      # leader re-sends un-ACKed commands [s]
    cmd_max_attempts: int = 60
    delayed_threshold_s: float = 0.15   # latency >= this is shown as DELAYED
    link_window: int = 40               # packets used for link-health window
    degraded_loss_rate: float = 0.20
    degraded_age_s: float = 0.60
    offline_timeout_s: float = 2.00
    comm_log_size: int = 300
    status_log_size: int = 100
    history_size: int = 100

    # ---- NEW: compatibility switch -----------------------------------------
    # True reproduces the original notebook behaviour exactly (no boundary
    # logic, instantaneous back-dated packets, constant feed-forward, ...).
    legacy_compat: bool = False

    # ------------------------------------------------------------------
    @classmethod
    def legacy(cls, **overrides) -> "Config":
        """Config that reproduces the original notebook (research mode)."""
        base = cls(
            legacy_compat=True,
            causal_delivery=False,
            boundary_guard=False,
            clamp_targets=False,
            clamp_to_world=False,
            dr_lambda=0.0,
            avoid_gain=0.25,
            avoid_swirl=0.0,
        )
        return replace(base, **overrides) if overrides else base

    def copy(self, **overrides) -> "Config":
        return replace(self, **overrides)

    @property
    def world(self) -> tuple:
        return (self.world_x_min, self.world_x_max,
                self.world_y_min, self.world_y_max)


# Module-level default instance (mirrors `CFG` in the notebook)
CFG = Config()
