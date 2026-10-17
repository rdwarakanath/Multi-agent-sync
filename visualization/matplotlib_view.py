"""
matplotlib_view.py -- live interactive view of the multi-agent simulation.

Design rules (they are what fixes the old 'robots disappear from the frame' bug)
-------------------------------------------------------------------------------
* FIXED FRAME   xlim / ylim / aspect are set once from the world bounds and
                autoscale is switched off.  The simulation itself keeps agents
                inside those bounds (boundary guard), so the view never has to
                'zoom out' to hide a problem.
* PERSISTENT ARTISTS   every robot, trail, target, link and text is created
                once in `_build()` and only updated with set_data / set_xy /
                set_text afterwards.  There is no `ax.clear()` per frame, so
                axis settings can never be rebuilt or lost.
* LIVE, NOT REPLAYED   the animation calls `sim.step()` itself, so the picture
                is always the current simulation state, not a precomputed array.
* PURE CLIENT   the view only uses the engine's public API:
                    sim.step()  sim.get_state()  sim.submit_command()
                    sim.handle_key()
                It never reads or writes agent state directly, so the same
                engine can be driven by the web dashboard or Webots unchanged.

Controls
--------
  W / S         accelerate / reverse the leader (switches to MANUAL mode)
  A / D         turn left / right
  SPACE         STOP        R  REGROUP      X  SCATTER      F  FOLLOW
  left-click    GOTO that point on the map
  buttons       FOLLOW  SCATTER  REGROUP  STOP  RESUME  ALLOCATE  MODE
  command box   any text command:  GOTO 10 5 | TASK A 4 7 | ALLOCATE | STATUS ...

Quick start (project root on the Python path)
---------------------------------------------
    python -m visualization.matplotlib_view            # desktop window

    from visualization import MatplotlibView
    view = MatplotlibView()
    view.show()                                         # desktop, interactive
    view.to_html(duration=20)                           # Jupyter inline playback

Interactive keyboard/mouse control in Jupyter needs the ipympl backend
(`%matplotlib widget`); with the default inline backend use `to_html`.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, TextBox

from simulation import MultiAgentSimulation, SimMode

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
LEADER_COLOR = "#d62728"
FOLLOWER_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
                   "#8c564b", "#e377c2", "#7f7f7f"]
PACKET_COLORS = {"OK": "#2ca02c", "DELAYED": "#ff9900", "LOST": "#d62728"}
LINK_COLORS = {"ONLINE": "#2ca02c", "DEGRADED": "#ff9900",
               "OFFLINE": "#d62728", "SELF": "#999999"}
TASK_COLORS = {"PENDING": "#bbbbbb", "ASSIGNED": "#ff9900",
               "IN_PROGRESS": "#f2c200", "DONE": "#2ca02c", "CANCELLED": "#dddddd"}
MODE_CYCLE = ["AUTO", "MANUAL", "COMMAND"]

# keys Matplotlib binds by default that collide with robot control
_STOLEN_KEYS = {"w", "a", "s", "d", "r", "x", "f", " ", "space",
                "left", "right", "up", "down"}
_ONE_SHOT = {"SPACE", "R", "X", "F"}


def _free_keymaps() -> None:
    """Remove Matplotlib's default shortcuts for the keys we use for driving."""
    for name in list(matplotlib.rcParams):
        if name.startswith("keymap."):
            matplotlib.rcParams[name] = [k for k in matplotlib.rcParams[name]
                                         if k not in _STOLEN_KEYS]


class MatplotlibView:
    def __init__(self, sim: Optional[MultiAgentSimulation] = None,
                 steps_per_frame: int = 2, speed: float = 1.0,
                 trail_len: int = 160, figsize=(16, 9),
                 show_slots: bool = True, show_comm_lines: bool = True):
        self.sim = sim or MultiAgentSimulation()
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.speed = max(0.05, float(speed))
        self.trail_len = trail_len
        self.show_slots = show_slots
        self.show_comm_lines = show_comm_lines
        self.figsize = figsize

        self._held: set = set()
        self._flash = ("", True, 0.0)          # (message, ok, expires at sim time)
        self._trails: Dict[int, Deque] = {}
        self._widgets: list = []
        self.anim: Optional[FuncAnimation] = None
        self._build()

    # ==================================================================
    # Construction (everything is created exactly once)
    # ==================================================================
    def _build(self) -> None:
        _free_keymaps()
        st = self.sim.get_state()
        w = st["world"]
        n = len(st["agents"])

        self.fig = plt.figure(figsize=self.figsize, facecolor="#f4f4f4")
        try:
            self.fig.canvas.manager.set_window_title("Multi-Agent Digital Twin")
        except Exception:
            pass

        # ---- world axes: FIXED limits, no autoscale ---------------------
        self.ax = self.fig.add_axes([0.04, 0.23, 0.56, 0.71])
        pad = 1.0
        self.ax.set_xlim(w["x_min"] - pad, w["x_max"] + pad)
        self.ax.set_ylim(w["y_min"] - pad, w["y_max"] + pad)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_autoscale_on(False)
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.add_patch(Rectangle((w["x_min"], w["y_min"]),
                                    w["x_max"] - w["x_min"], w["y_max"] - w["y_min"],
                                    fill=False, ec="black", lw=2.0, zorder=1))
        m = self.sim.cfg.boundary_margin
        self.ax.add_patch(Rectangle((w["x_min"] + m, w["y_min"] + m),
                                    w["x_max"] - w["x_min"] - 2 * m,
                                    w["y_max"] - w["y_min"] - 2 * m,
                                    fill=False, ec="#999999", lw=0.8, ls="--", zorder=1))
        self.title = self.ax.set_title("", fontsize=10, family="monospace")

        # ---- per-agent artists -------------------------------------------
        self.bodies: List[Polygon] = []
        self.trail_lines = []
        self.labels = []
        self.target_links = []
        self.comm_lines = []
        for i in range(n):
            color = LEADER_COLOR if i == 0 else FOLLOWER_COLORS[(i - 1) % len(FOLLOWER_COLORS)]
            self._trails[i] = deque(maxlen=self.trail_len)
            (tl,) = self.ax.plot([], [], "-", color=color, alpha=0.45, lw=1.2, zorder=2)
            body = Polygon(np.zeros((3, 2)), closed=True, fc=color, ec="black",
                           lw=1.0, zorder=6)
            self.ax.add_patch(body)
            lab = self.ax.text(0, 0, "", fontsize=8, weight="bold", zorder=7,
                               ha="center", va="bottom")
            (lnk,) = self.ax.plot([], [], ":", color=color, alpha=0.7, lw=1.0, zorder=3)
            self.trail_lines.append(tl)
            self.bodies.append(body)
            self.labels.append(lab)
            self.target_links.append(lnk)
            if i > 0:
                (cl,) = self.ax.plot([], [], "-", color="#999999", lw=1.4,
                                     alpha=0.55, zorder=2)
                self.comm_lines.append(cl)

        # ---- targets / slots / tasks ---------------------------------------
        (self.slots,) = self.ax.plot([], [], "o", mfc="none", mec="#888888",
                                     ms=10, mew=1.2, zorder=3, ls="none")
        (self.goto_marker,) = self.ax.plot([], [], "*", color="#e41a1c", ms=18,
                                           mec="black", zorder=4, ls="none")
        (self.scatter_marker,) = self.ax.plot([], [], "X", color="#984ea3", ms=11,
                                              mec="black", zorder=4, ls="none")
        (self.pending_marker,) = self.ax.plot([], [], "x", color="#888888", ms=10,
                                              mew=2, zorder=4, ls="none")
        self.task_scatter = self.ax.scatter([], [], marker="s", s=170, ec="black",
                                            zorder=4)
        self.task_labels: Dict[str, plt.Text] = {}
        self.halt_text = self.ax.text(0.5, 0.5, "", transform=self.ax.transAxes,
                                      ha="center", va="center", fontsize=16,
                                      color="white", weight="bold", zorder=20,
                                      bbox=dict(fc="#c0392b", ec="black", pad=8))
        self.halt_text.set_visible(False)

        # ---- side panels ----------------------------------------------------
        self.p_agents = self._panel([0.62, 0.745, 0.36, 0.195], "AGENTS")
        self.p_net = self._panel([0.62, 0.625, 0.36, 0.105], "NETWORK")
        self.p_tasks = self._panel([0.62, 0.525, 0.36, 0.085], "TASK ALLOCATION")
        self.p_comm = self._panel([0.62, 0.315, 0.36, 0.195], "COMMUNICATION LOG (real network events)")
        self.p_cmd = self._panel([0.62, 0.14, 0.36, 0.16], "COMMAND HISTORY")
        self.p_key = self._panel([0.62, 0.035, 0.36, 0.09], "KEYBOARD / STATUS")

        # ---- feedback line + widgets ----------------------------------------
        self.feedback = self.fig.text(0.04, 0.197, "", fontsize=9, family="monospace")
        labels = ["FOLLOW", "SCATTER", "REGROUP", "STOP", "RESUME", "ALLOCATE", "MODE: AUTO"]
        bw, gap, x0 = 0.075, 0.0065, 0.04
        self.buttons: Dict[str, Button] = {}
        for k, lab in enumerate(labels):
            bax = self.fig.add_axes([x0 + k * (bw + gap), 0.115, bw, 0.05])
            btn = Button(bax, lab, color="#e8eef7", hovercolor="#c9d9f0")
            btn.label.set_fontsize(8)
            self.buttons[lab.split(":")[0]] = btn
            self._widgets.append(btn)
        self.buttons["FOLLOW"].on_clicked(lambda e: self._run_cmd("FOLLOW"))
        self.buttons["SCATTER"].on_clicked(lambda e: self._run_cmd("SCATTER"))
        self.buttons["REGROUP"].on_clicked(lambda e: self._run_cmd("REGROUP"))
        self.buttons["STOP"].on_clicked(lambda e: self._run_cmd("STOP"))
        self.buttons["RESUME"].on_clicked(lambda e: self._run_cmd("RESUME"))
        self.buttons["ALLOCATE"].on_clicked(lambda e: self._run_cmd("ALLOCATE"))
        self.buttons["MODE"].on_clicked(self._cycle_mode)
        self.buttons["STOP"].ax.set_facecolor("#f7d6d6")

        tax = self.fig.add_axes([0.115, 0.04, 0.485, 0.05])
        self.textbox = TextBox(tax, "Command ", initial="", color="white",
                               hovercolor="#fffbe6")
        self.textbox.on_submit(self._on_textbox)
        self._widgets.append(self.textbox)

        # ---- events -----------------------------------------------------------
        c = self.fig.canvas
        c.mpl_connect("key_press_event", self._on_key_press)
        c.mpl_connect("key_release_event", self._on_key_release)
        c.mpl_connect("button_press_event", self._on_click)

        self.render()

    def _panel(self, rect, title: str):
        ax = self.fig.add_axes(rect, facecolor="white")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#bbbbbb")
        ax.text(0.01, 1.02, title, transform=ax.transAxes, fontsize=8,
                weight="bold", color="#444444", va="bottom")
        txt = ax.text(0.015, 0.96, "", transform=ax.transAxes, fontsize=8.2,
                      family="monospace", va="top", ha="left", linespacing=1.25)
        return txt

    # ==================================================================
    # Rendering (updates existing artists only)
    # ==================================================================
    def render(self) -> None:
        """Draw the CURRENT simulation state (does not advance the simulation)."""
        st = self.sim.get_state()
        self._update_agents(st)
        self._update_targets(st)
        self._update_panels(st)
        self._update_chrome(st)

    @staticmethod
    def _glyph(x, y, th, scale=1.0):
        pts = np.array([[0.50, 0.0], [-0.34, 0.30], [-0.20, 0.0], [-0.34, -0.30]]) * scale
        c, s = np.cos(th), np.sin(th)
        return pts @ np.array([[c, s], [-s, c]]) + [x, y]

    def _update_agents(self, st: dict) -> None:
        agents = st["agents"]
        lead = agents[0]
        last_pose = {}                       # latest POSE_UPDATE outcome per follower
        for e in st["messages"]:
            if e["type"] == "POSE_UPDATE":
                last_pose[e["receiver"]] = e["status"]
        for a in agents:
            i = a["id"]
            self.bodies[i].set_xy(self._glyph(a["x"], a["y"], a["theta"]))
            self._trails[i].append((a["x"], a["y"]))
            tx, ty = zip(*self._trails[i])
            self.trail_lines[i].set_data(tx, ty)
            self.labels[i].set_position((a["x"], a["y"] + 0.55))
            self.labels[i].set_text(a["short"])
            if a["target"] is not None:      # dotted line agent -> its target
                self.target_links[i].set_data([a["x"], a["target"][0]], [a["y"], a["target"][1]])
            else:
                self.target_links[i].set_data([], [])
            if i > 0:                        # live radio link colouring
                cl = self.comm_lines[i - 1]
                cl.set_visible(self.show_comm_lines)
                cl.set_data([lead["x"], a["x"]], [lead["y"], a["y"]])
                if a["link"] == "OFFLINE":
                    cl.set_color(LINK_COLORS["OFFLINE"]); cl.set_linestyle(":")
                else:
                    cl.set_color(PACKET_COLORS.get(last_pose.get(i, "OK"), "#999999"))
                    cl.set_linestyle("--" if a["link"] == "DEGRADED" else "-")

    def _update_targets(self, st: dict) -> None:
        pick = lambda kinds: [(t["x"], t["y"]) for t in st["targets"] if t["kind"] in kinds]

        def put(line, pts):
            if pts:
                xs, ys = zip(*pts)
                line.set_data(xs, ys)
            else:
                line.set_data([], [])

        put(self.goto_marker, pick(("goto",)))
        put(self.scatter_marker, pick(("scatter",)))
        put(self.pending_marker, pick(("scatter_pending",)))
        put(self.slots, [tuple(a["slot"]) for a in st["agents"]
                         if a["slot"] is not None] if self.show_slots else [])

        tasks = st["tasks"]
        if tasks:
            self.task_scatter.set_offsets([[t["x"], t["y"]] for t in tasks])
            self.task_scatter.set_facecolors([TASK_COLORS.get(t["status"], "#bbbbbb") for t in tasks])
        else:
            self.task_scatter.set_offsets(np.empty((0, 2)))
        seen = set()
        for t in tasks:
            seen.add(t["id"])
            lab = self.task_labels.get(t["id"])
            if lab is None:
                lab = self.ax.text(0, 0, "", fontsize=9, weight="bold", ha="center",
                                   va="center", zorder=5, color="#222222")
                self.task_labels[t["id"]] = lab
            lab.set_position((t["x"], t["y"] - 0.75))
            lab.set_text(t["id"])
            lab.set_visible(True)
        for tid, lab in self.task_labels.items():
            if tid not in seen:
                lab.set_visible(False)

    def _update_panels(self, st: dict) -> None:
        # agents
        rows = [f"{'ID':<3} {'MODE':<8} {'X':>6} {'Y':>6} {'V':>5} {'LINK':<8} TASK"]
        for a in st["agents"]:
            rows.append(f"{a['short']:<3} {a['mode']:<8} {a['x']:6.1f} {a['y']:6.1f} "
                        f"{a['v']:5.2f} {a['link']:<8} {a['task'] or ''}")
        self.p_agents.set_text("\n".join(rows))

        # network
        net = st["network"]
        flag = "   ** FORCED LOSS **" if net["forced_loss"] else ""
        self.p_net.set_text(
            f"packets sent {net['sent']:>6}   received {net['received']:>6}   lost {net['lost']:>5}\n"
            f"loss rate {net['loss_rate']:5.1f} %   latency avg {1000 * net['avg_latency']:4.0f} ms"
            f"  max {1000 * net['max_latency']:4.0f} ms\n"
            f"commands awaiting ACK: {st['pending_commands']}   boundary clamps: {st['boundary_clamps']}{flag}")

        # tasks
        self.p_tasks.set_text("\n".join(st["allocation"][-4:]) or "(no tasks)")

        # communication log: real events, newest last
        ev = [e for e in st["messages"] if e["type"] != "POSE_UPDATE"][-9:]
        pose = [e for e in st["messages"] if e["type"] == "POSE_UPDATE"]
        lines = []
        for e in ev:
            tail = e["status"] if e["status"] != "DELAYED" else f"DELAY={e['latency']:.2f}s"
            lines.append(f"{e['t']:7.2f}  {e['from']:>2}->{e['to']:<2}  {e['type']:<11} {tail}")
        if pose:
            ok = sum(1 for e in pose if e["status"] != "LOST")
            lines.append(f"-- POSE_UPDATE stream (last {len(pose)}): {ok} ok / {len(pose) - ok} lost")
        self.p_comm.set_text("\n".join(lines) or "(no events yet)")

        # command history
        hist = []
        for h in st["command_history"][:7]:
            mark = "ok " if h["ok"] else "ERR"
            hist.append(f"[{h['clock']}] {mark} {h['text'][:18]:<18} {h['message'][:34]}")
        self.p_cmd.set_text("\n".join(hist) or "(no commands yet)")

        # keyboard + status
        kb = st["keyboard"]
        keys = "  ".join(f"{k}[{'#' if v == 'PRESSED' else ' '}]" for k, v in kb.items())
        mc = st["manual_command"]
        recent = st["status_log"][-1]["text"] if st["status_log"] else ""
        self.p_key.set_text(f"{keys}   v_cmd={mc['v']:+.2f}  w_cmd={mc['omega']:+.2f}\n"
                            f"mode={st['sim_mode']}  mission={st['mission_state']}\n"
                            f"last: {recent[:52]}")

    def _update_chrome(self, st: dict) -> None:
        rms = st["formation_rms"]
        sep = st["min_distance"]
        self.title.set_text(
            f"t={st['time']:6.1f}s | MODE {st['sim_mode']} | MISSION {st['mission_state']}"
            f" | formation {st['formation']['name']}"
            + ("" if rms is None else f" | form.err {rms:.2f} m | min sep {sep:.2f} m"))
        self.buttons["MODE"].label.set_text(f"MODE: {st['sim_mode']}")
        msg, ok, until = self._flash
        if msg and st["time"] <= until:
            self.feedback.set_text(msg[:110])
            self.feedback.set_color("#1b7f2a" if ok else "#c0392b")
        else:
            self.feedback.set_text("")
        self.halt_text.set_visible(bool(st["halted"]))
        if st["halted"]:
            self.halt_text.set_text(f"SIMULATION HALTED\n{st['fault']}")

    # ==================================================================
    # Input handling (all of it goes through the engine's public API)
    # ==================================================================
    def _run_cmd(self, text: str) -> None:
        res = self.sim.submit_command(text)
        self._flash = (f"> {text}: {res.message}", res.ok, self.sim.t + 6.0)

    def _on_textbox(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._run_cmd(text)
        self.textbox.set_val("")

    def _cycle_mode(self, _event) -> None:
        cur = self.sim.get_state()["sim_mode"]
        nxt = MODE_CYCLE[(MODE_CYCLE.index(cur) + 1) % len(MODE_CYCLE)]
        self._run_cmd(f"MODE {nxt}")

    def _typing(self) -> bool:
        return bool(getattr(self.textbox, "capturekeystrokes", False))

    def _on_key_press(self, event) -> None:
        if self._typing():                       # user is typing in the command box
            return
        k = self.sim.keyboard.normalize(event.key)
        if k is None:
            return
        if k in _ONE_SHOT:                       # ignore auto-repeat of one-shots
            if k in self._held:
                return
            self._held.add(k)
        self.sim.handle_key(event.key, True)

    def _on_key_release(self, event) -> None:
        k = self.sim.keyboard.normalize(event.key)
        if k is None:
            return
        self._held.discard(k)
        self.sim.handle_key(event.key, False)

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1 or event.xdata is None:
            return
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):   # zoom/pan active
            return
        self._run_cmd(f"GOTO {event.xdata:.2f} {event.ydata:.2f}")

    # ==================================================================
    # Running
    # ==================================================================
    def _frame(self, _i) -> list:
        for _ in range(self.steps_per_frame):
            if not self.sim.step():
                break
        self.render()
        return []

    def animate(self) -> FuncAnimation:
        """Create the live animation (does not open a window)."""
        interval = max(10, int(1000 * self.sim.dt * self.steps_per_frame / self.speed))
        self.anim = FuncAnimation(self.fig, self._frame, interval=interval,
                                  blit=False, cache_frame_data=False)
        return self.anim

    def show(self) -> None:
        """Open the interactive window (blocking on desktop backends)."""
        self.animate()
        plt.show()

    def run_steps(self, n_frames: int) -> None:
        """Advance + render without a window (tests, screenshots)."""
        for k in range(n_frames):
            self._frame(k)

    def save_frame(self, path: str, dpi: int = 100) -> str:
        self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())
        return path

    def save_animation(self, path: str, duration: float = 10.0, fps: int = 10) -> str:
        """Render a GIF/MP4 of the running simulation (e.g. for reports)."""
        frames = int(duration * fps)
        self.steps_per_frame = max(1, int(round(1.0 / (fps * self.sim.dt))))
        anim = FuncAnimation(self.fig, self._frame, frames=frames, blit=False,
                             cache_frame_data=False)
        anim.save(path, fps=fps)
        return path

    def to_html(self, duration: float = 15.0, fps: int = 10):
        """Jupyter (inline backend): scripted playback as an HTML5 animation."""
        from IPython.display import HTML
        frames = int(duration * fps)
        self.steps_per_frame = max(1, int(round(1.0 / (fps * self.sim.dt))))
        anim = FuncAnimation(self.fig, self._frame, frames=frames, blit=False,
                             cache_frame_data=False)
        html = HTML(anim.to_jshtml(fps=fps))
        plt.close(self.fig)
        return html


def main() -> None:
    """Desktop entry point:  python -m visualization.matplotlib_view"""
    view = MatplotlibView(MultiAgentSimulation(sim_mode=SimMode.AUTO))
    view.show()


if __name__ == "__main__":
    main()
