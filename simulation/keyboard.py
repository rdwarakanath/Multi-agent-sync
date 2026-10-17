"""
keyboard.py -- keyboard state -> desired leader command.

This module is UI-independent: any front end (Matplotlib key events, the
browser's keydown/keyup, a CLI) just calls `press(key)` / `release(key)`.

Keyboard input NEVER touches robot state.  It only produces a desired
(v_cmd, omega_cmd) which the simulation passes through the normal
boundary guard -> actuator lag -> RK4 plant (spec section 8).

Keys
----
W / S      accelerate forward / reverse  (hold)   -> v_cmd ramps up/down
A / D      turn left / right             (hold)   -> omega_cmd ramps
SPACE      stop (one-shot)
R          regroup (one-shot)      X   scatter (one-shot)      F   follow (one-shot)

Releasing W/S lets v_cmd coast back to zero; releasing A/D straightens out.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from .config import Config

_ALIASES = {
    "w": "W", "arrowup": "W", "up": "W",
    "s": "S", "arrowdown": "S", "down": "S",
    "a": "A", "arrowleft": "A", "left": "A",
    "d": "D", "arrowright": "D", "right": "D",
    " ": "SPACE", "space": "SPACE", "spacebar": "SPACE",
    "r": "R", "x": "X", "f": "F",
}
_DRIVE_KEYS = ("W", "A", "S", "D")
# one-shot keys -> command text executed by the simulation
ONE_SHOT = {"SPACE": "STOP", "R": "REGROUP", "X": "SCATTER", "F": "FOLLOW"}


class KeyboardController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pressed: Dict[str, bool] = {k: False for k in _DRIVE_KEYS}
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self._events: Deque[str] = deque()

    # ------------------------------------------------------------------
    @staticmethod
    def normalize(key: str) -> Optional[str]:
        """Map any key spelling ('w', 'W', 'ArrowUp', ' ') to a canonical name."""
        if key is None:
            return None
        return _ALIASES.get(str(key).lower())

    def press(self, key: str) -> Optional[str]:
        k = self.normalize(key)
        if k is None:
            return None
        if k in self.pressed:
            self.pressed[k] = True
        elif k == "SPACE":
            self.v_cmd = self.w_cmd = 0.0          # immediate zero command
            self._events.append(ONE_SHOT[k])
        else:
            self._events.append(ONE_SHOT[k])
        return k

    def release(self, key: str) -> Optional[str]:
        k = self.normalize(key)
        if k in self.pressed:
            self.pressed[k] = False
        return k

    def set_key(self, key: str, down: bool) -> Optional[str]:
        return self.press(key) if down else self.release(key)

    def release_all(self) -> None:
        for k in self.pressed:
            self.pressed[k] = False

    def reset(self) -> None:
        self.release_all()
        self.v_cmd = self.w_cmd = 0.0
        self._events.clear()

    @property
    def driving(self) -> bool:
        """True while any W/A/S/D key is held (or the command is still coasting)."""
        return any(self.pressed.values()) or abs(self.v_cmd) > 1e-3 or abs(self.w_cmd) > 1e-3

    # ------------------------------------------------------------------
    def command(self, dt: float) -> Tuple[float, float]:
        """Advance the ramps by dt and return the desired (v_cmd, omega_cmd)."""
        c = self.cfg
        fwd, back = self.pressed["W"], self.pressed["S"]
        if fwd and not back:
            self.v_cmd = min(self.v_cmd + c.manual_accel_v * dt, c.manual_max_v)
        elif back and not fwd:
            self.v_cmd = max(self.v_cmd - c.manual_accel_v * dt, -c.manual_max_reverse)
        else:                                           # coast to zero
            step = c.manual_decay_v * dt
            self.v_cmd = 0.0 if abs(self.v_cmd) <= step else self.v_cmd - step * (1 if self.v_cmd > 0 else -1)

        left, right = self.pressed["A"], self.pressed["D"]
        target_w = 0.0
        if left and not right:
            target_w = c.manual_max_w                   # +omega = counter-clockwise
        elif right and not left:
            target_w = -c.manual_max_w
        rate = (c.manual_accel_w if target_w != 0.0 else c.manual_decay_w) * dt
        diff = target_w - self.w_cmd
        self.w_cmd += max(-rate, min(rate, diff))
        return self.v_cmd, self.w_cmd

    def consume_events(self) -> List[str]:
        """Pop pending one-shot command texts (STOP / REGROUP / SCATTER / FOLLOW)."""
        out = list(self._events)
        self._events.clear()
        return out

    def state_report(self) -> Dict[str, str]:
        """For the UI:  {'W': 'PRESSED', 'A': 'RELEASED', ...}"""
        return {k: ("PRESSED" if v else "RELEASED") for k, v in self.pressed.items()}
