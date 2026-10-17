"""
metrics.py -- performance metrics and analysis helpers (RESEARCH MODE).

Everything the original notebook computed lives here as reusable functions:

  compute_metrics        ITAE, RMSE, peak error, effort, min separation, loss, collisions
  routh_hurwitz_cubic    stability test for the inner velocity loop (Eq. 22)
  stability_scan         Kp/Kd grid screening + best-margin pick
  robustness_sweep       formation error vs packet-loss scale
  monte_carlo            randomised loss + sensor-noise reliability study
  summarize_monte_carlo  mean / 95th percentile / worst-case statistics

The sweep functions take a `simulate_fn(mode, seed, loss_scale, sensor_sigma)`
(e.g. `simulation.simulation.simulate`) so this module does not import the
engine and stays free of circular dependencies.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np

from .config import CFG, Config

_trapz = getattr(np, "trapezoid", None) or np.trapz     # NumPy 1.x / 2.x


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------
def compute_metrics(h: dict) -> Dict[str, float]:
    """Metrics of one run history (dict as returned by `simulate`)."""
    t, e = h["t"], h["formation_rms"]
    return dict(
        ITAE=float(_trapz(t * e, t)),
        RMSE=float(np.sqrt(np.mean(e ** 2))),
        Peak=float(np.max(e)),
        Effort=float(np.mean(h["control_effort"])),
        MinSep=float(np.min(h["min_distance"])),
        Loss=float(np.mean(h["packet_drop"]) * 100),
        Collision=float(np.mean(h["collision"]) * 100),
    )


def improvement_pct(baseline: float, other: float) -> float:
    """Percentage improvement of `other` over `baseline` (positive = better)."""
    return 100.0 * (baseline - other) / baseline if baseline else 0.0


# ---------------------------------------------------------------------------
# Routh-Hurwitz gain screening (Eq. 21-22 of the report)
# ---------------------------------------------------------------------------
def routh_hurwitz_cubic(a3, a2, a1, a0):
    """Stability of  a3 s^3 + a2 s^2 + a1 s + a0.
    Returns (stable: bool, margin: float)."""
    stable = (a3 > 0 and a2 > 0 and a1 > 0 and a0 > 0 and a2 * a1 > a3 * a0)
    return stable, min(a3, a2, a1, a0, a2 * a1 - a3 * a0)


def pid_coefficients(kp, ki, kd, tau=CFG.tau_motor):
    """Characteristic-polynomial coefficients (as used in the notebook)."""
    return tau, 1 + tau * kd, kp + kd, ki


def stability_scan(cfg: Config | None = None, n: int = 16,
                   kp_range=(0.5, 3.2), kd_range=(0.15, 1.5)) -> dict:
    """Grid-screen (Kp, Kd) with Ki fixed; pick the stable point with the
    largest Routh-Hurwitz margin."""
    cfg = cfg or CFG
    kp_grid = np.linspace(*kp_range, n)
    kd_grid = np.linspace(*kd_range, n)
    stable = np.zeros((n, n))
    margin = np.zeros((n, n))
    for a, kp in enumerate(kp_grid):
        for b, kd in enumerate(kd_grid):
            stable[a, b], margin[a, b] = routh_hurwitz_cubic(
                *pid_coefficients(kp, cfg.ki, kd, cfg.tau_motor))
    idx = np.unravel_index(np.argmax(np.where(stable > 0, margin, -np.inf)),
                           margin.shape)
    return dict(kp_grid=kp_grid, kd_grid=kd_grid, stable_map=stable,
                margin_map=margin, best_kp=float(kp_grid[idx[0]]),
                best_kd=float(kd_grid[idx[1]]), ki=cfg.ki)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
SimulateFn = Callable[..., dict]


def robustness_sweep(simulate_fn: SimulateFn,
                     loss_levels: Iterable[float] = (0, .5, 1, 1.5, 2, 2.5, 3),
                     runs_per_level: int = 5, base_seed: int = 300,
                     mode: str = "advanced") -> np.ndarray:
    """Columns: [RMSE, peak error, min separation, packet loss %] per level."""
    rows = []
    for scale in loss_levels:
        vals = []
        for s in range(runs_per_level):
            h = simulate_fn(mode, base_seed + s, loss_scale=scale)
            vals.append([np.sqrt(np.mean(h["formation_rms"] ** 2)),
                         np.max(h["formation_rms"]),
                         np.min(h["min_distance"]),
                         np.mean(h["packet_drop"]) * 100])
        rows.append(np.mean(vals, axis=0))
    return np.array(rows)


def monte_carlo(simulate_fn: SimulateFn, runs: int = 30,
                mode: str = "advanced", base_seed: int = 1000) -> np.ndarray:
    """Randomised loss scale and sensor noise.  Columns:
    [RMSE, peak, min separation, packet loss %, control effort]."""
    out = []
    for s in range(runs):
        loss = np.random.default_rng(5000 + s).uniform(.6, 2.8)
        sigma = np.random.default_rng(6000 + s).uniform(.015, .07)
        h = simulate_fn(mode, base_seed + s, loss_scale=loss, sensor_sigma=sigma)
        out.append([np.sqrt(np.mean(h["formation_rms"] ** 2)),
                    np.max(h["formation_rms"]),
                    np.min(h["min_distance"]),
                    np.mean(h["packet_drop"]) * 100,
                    np.mean(h["control_effort"])])
    return np.array(out)


def summarize_monte_carlo(results: np.ndarray) -> Dict[str, float]:
    return dict(
        mean_rmse=float(results[:, 0].mean()),
        p95_rmse=float(np.percentile(results[:, 0], 95)),
        worst_peak=float(results[:, 1].max()),
        worst_min_separation=float(results[:, 2].min()),
        mean_packet_loss=float(results[:, 3].mean()),
        mean_effort=float(results[:, 4].mean()),
    )
