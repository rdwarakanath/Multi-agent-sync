# Multi-Agent Digital Twin

An interactive, software-only digital twin of a **leader–follower swarm of
differential-drive robots** that talk over a **lossy wireless link** (an
ESP-NOW stand-in). One leader and four followers (configurable) hold a
formation, react to operator commands, share work through task allocation, and
keep working when radio packets are lost.

It is built as a reusable **simulation engine** with three front ends on top:

| Front end | Purpose |
|---|---|
| **Python / Jupyter API** | Research mode: batch runs, ZOH vs PIDR vs Advanced comparison, Monte-Carlo, stability analysis |
| **Matplotlib window** | Local interactive view with keyboard, mouse and buttons |
| **Web dashboard (FastAPI + HTML)** | Browser control panel with live map, commands, tasks and communication log |

Pure Python. No Webots, ROS 2 or hardware is required (the architecture is
ready for them, see [Roadmap](#roadmap)).

---

## Contents

1. [Features](#features)
2. [Install](#install)
3. [Quick start](#quick-start)
4. [The web dashboard: what every part does](#the-web-dashboard-what-every-part-does)
5. [The Matplotlib window](#the-matplotlib-window)
6. [Commands reference](#commands-reference)
7. [Keyboard control](#keyboard-control)
8. [Modes and mission states](#modes-and-mission-states)
9. [How it works](#how-it-works)
10. [Project structure](#project-structure)
11. [Python API](#python-api)
12. [REST / WebSocket API](#rest--websocket-api)
13. [Configuration](#configuration)
14. [Logging and metrics](#logging-and-metrics)
15. [Research mode (notebook compatibility)](#research-mode-notebook-compatibility)
16. [Troubleshooting](#troubleshooting)
17. [Known limitations](#known-limitations)
18. [Roadmap](#roadmap)

---

## Features

**Swarm behaviour**
- Leader plus N followers (default 5 agents) with a wedge formation. LINE and COLUMN presets are also built in.
- Followers track their formation slot using a dead-reckoned estimate of the leader, a cascaded PID controller with adaptive gains, neighbour coupling and collision avoidance.
- `GOTO`, `MOVE`, `SCATTER`, `REGROUP`, `FOLLOW`, `STOP`, `RESUME`, `FORMATION`, `STATUS`.
- Task allocation: create tasks, allocate them to the nearest free follower by priority, watch them complete, and the swarm regroups automatically afterwards.
- Single-agent commands, for example `AGENT 2 GOTO 10 5` or `AGENT 3 STOP`.

**Communication layer (real, not cosmetic)**
- Every leader→follower and follower→leader message passes through a Gilbert–Elliott burst-loss channel with random latency.
- Message types: `POSE_UPDATE`, `COMMAND`, `TASK_ASSIGN`, `SCATTER`, `REGROUP`, `FOLLOW`, `STOP`, `ACK`, `HEARTBEAT`.
- Commands are **reliable**: they carry an id, are acknowledged, and are re-sent until acknowledged, so a lost `SCATTER` still arrives eventually.
- Live link health per follower (`ONLINE`, `DEGRADED`, `OFFLINE`), per-link statistics, and a communication log that only shows what the network model actually did.
- Force 100 % packet loss to test dead-reckoning behaviour.

**Safety**
- The world is a bounded rectangle. Robots cannot leave it: targets are validated and clamped, a soft boundary guard steers robots away from walls, and a hard clamp is a last-resort numerical guard.
- Inter-agent avoidance, a "yield to whoever is ahead" speed rule, and a rigid-formation shift near walls so the formation does not collapse into a pile.
- NaN / Inf detection: the simulation halts and reports which agent failed instead of corrupting the display.
- Bad commands never crash anything; they return a clear message.

**Interfaces**
- Keyboard driving of the leader (W/A/S/D, SPACE, R, X, F).
- Click-on-map GOTO.
- Command history, status events, and CSV logging.

---

## Install

Python 3.10 or newer.

```bash
cd multi_agent_digital_twin
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` contains numpy, scipy, matplotlib, pandas, fastapi,
uvicorn, jinja2 and httpx (httpx is only needed for API tests).

Run every command below **from the project root** (the folder that contains
`simulation/`, `visualization/` and `web/`) so the packages can be imported.

---

## Quick start

### 1. Web dashboard (recommended)

```bash
python -m web.app
```

Open <http://127.0.0.1:8000>. Options:

```bash
python -m web.app --host 0.0.0.0 --port 8080 --seed 7
uvicorn web.app:app --reload        # development auto-reload
```

### 2. Matplotlib window

```bash
python -m visualization.matplotlib_view
```

### 3. Headless / scripting

```python
from simulation import MultiAgentSimulation

sim = MultiAgentSimulation()
sim.submit_command("GOTO 10 5")
sim.run(20)                      # simulate 20 seconds
print(sim.status_text())
```

### 4. Jupyter

```python
from visualization import MatplotlibView
view = MatplotlibView()
view.to_html(duration=20)        # inline playback
```

For live keyboard and mouse control inside Jupyter, enable the ipympl backend
first (`%matplotlib widget`).

---

## The web dashboard: what every part does

The page has no tab bar. It is a single control room split into a **top bar**,
a **left side** (map and controls) and a **right side** (information panels).
Everything you click is sent to the server, validated by the engine, and the
result shows up in the feedback line, the panels, and on the map.

### Top bar

| Item | What it does |
|---|---|
| **t = … s** | Simulation time in seconds. |
| **MODE …** | How the leader is driven right now: `AUTO`, `MANUAL` or `COMMAND` (see [Modes](#modes-and-mission-states)). |
| **MISSION …** | What the swarm is doing: `FOLLOWING`, `GOTO`, `SCATTERING`, `REGROUPING`, `TASK_EXECUTION`, `STOPPED`. |
| **FORMATION …** | Active formation preset (`WEDGE`, `LINE`, `COLUMN`). |
| **live / reconnecting…** | Connection to the server. Green "live" means the WebSocket is streaming. If it drops, the page falls back to polling and reconnects automatically. |
| **Pause / Resume clock** | Freezes or resumes the simulation clock on the server. Commands can still be sent while paused. |
| **Speed** | Simulation speed: 0.5x, 1x, 2x or 4x real time. |
| **Control** | Controller used by followers: `advanced` (adaptive gains, neighbour coupling, avoidance), `pidr` (PID plus dead reckoning, no avoidance) or `zoh` (plain hold of the last packet). Takes effect on **Reset**. |
| **Loss x** | Packet-loss multiplier (1 = nominal, 3 = heavy loss, 0 = none). Takes effect on **Reset**. |
| **Reset** | Restarts the simulation with the Control, Loss and Mode settings currently selected. Trails are cleared. |

### Left side: the map

The large canvas is the live world. The solid black rectangle is the world
boundary; the dashed rectangle inside it marks the zone where the soft
boundary guard starts steering robots back. The view is fixed and never
rescales, because robots physically cannot leave the world.

| Drawn item | Meaning |
|---|---|
| **Coloured arrowhead** | A robot. Red = leader (`L`), others = followers (`F1`, `F2`, …). The label sits above it. A **red outline** means that robot is `STOPPED`. |
| **Faint coloured line behind a robot** | Its recent path (trail). |
| **Line from leader to each follower** | The live radio link. Colour shows what the network did with the latest position packet: **green** delivered, **orange** delayed, **red** lost. A **dashed** line means the link is `DEGRADED`; a **dotted red** line means `OFFLINE`. |
| **Hollow grey circles** | Formation slots, the places followers are trying to reach. |
| **Red star** | An active `GOTO` target. |
| **Purple X** | A scatter target a follower is heading to. |
| **Grey x** | A scatter target that has been commanded but not yet delivered to the follower (the command is still in flight or being retried). |
| **Dotted coloured line from a robot to a point** | The robot's own current target. |
| **Squares with a letter** | Tasks. Grey = pending, orange = assigned, yellow = in progress, green = done. The letter below is the task id. |

**Clicking the map** sends the agent chosen in *Target agent* to that point
(`GOTO x y`). With *Target agent = ALL* it moves the leader and the
formation follows.

Under the map, a one-line **feedback bar** shows the result of your last
action: green for success, red for an error or rejection.

### Left side: the control panel

**Row 1: swarm commands**

| Control | What it does |
|---|---|
| **Target agent** | Who `STOP`, `RESUME` and map-click `GOTO` apply to: `ALL`, `LEADER`, or one follower (`Agent 1` … ). `GOTO` with `ALL` or `LEADER` moves the leader. `STOP` / `RESUME` with `ALL` or `LEADER` act on the whole swarm; only a single follower can be stopped on its own. |
| **FOLLOW** | Normal leader–follower behaviour: followers hold formation slots. Also cancels active tasks. |
| **SCATTER** | Followers disperse to separate safe points around the leader, then hold. The swarm logs `SCATTER COMPLETE`. |
| **REGROUP** | Followers return to their formation slots and the swarm logs `REGROUP COMPLETE`, then returns to `FOLLOWING`. |
| **STOP** | Everything stops safely (the current mission is remembered). With a single follower selected in *Target agent*, only that follower stops. |
| **RESUME** | Continues the mission that `STOP` interrupted. |
| **Mode** | Switches the leader between `AUTO`, `MANUAL`, `COMMAND`. |
| **Formation** | Switches the formation to `WEDGE`, `LINE` or `COLUMN` and regroups into it. |

**Row 2: precise input**

| Control | What it does |
|---|---|
| **GOTO x y → Go** | Sends the selected agent to a world coordinate. Targets outside the world are clamped inside and the feedback bar tells you; non-numeric values are rejected. |
| **Command box → Send** | Accepts any [text command](#commands-reference), for example `MOVE 3 -2`, `AGENT 2 STOP`, `FORMATION COLUMN`, `STATUS`. Unknown commands return `UNKNOWN COMMAND`. |

**Row 3: keyboard and fault testing**

| Control | What it does |
|---|---|
| **Key indicators (W A S D)** | Light up while the server sees the key as pressed. If a key stays lit, click elsewhere or press and release it. |
| **v_cmd / w_cmd** | The speed and turn-rate command that keyboard driving is currently producing. |
| **force 100% packet loss** | Switches every radio link to total loss. Use it to watch dead reckoning take over: links go `OFFLINE`, lines turn red, and followers keep moving on the last known leader motion, which fades out over time. Untick to restore the link. |

### Right side: information panels

**Agents.** One row per robot: ID (with its colour), current **Mode**
(`AUTO`, `MANUAL`, `GOTO`, `FOLLOW`, `SCATTER`, `REGROUP`, `TASK`, `STOPPED`,
`IDLE`), **X**, **Y**, speed **V**, radio **Link** status, and assigned
**Task**. The mode column is each follower's own view, so after a command it
only changes once the command message actually arrives.

**Network.** Live statistics from the channel model:

- *sent / received / lost* and *loss rate*: all traffic, not just positions
- *latency avg / max*
- *awaiting ACK*: commands still being retried because no acknowledgement arrived
- *boundary clamps*: how often the hard last-resort clamp had to fire (normally 0)
- *form. error*: how far followers are from their slots (RMS, metres)
- *min separation*: the smallest distance between any two robots

**Task allocation.** Create and manage tasks:

- Fill in **id**, **x**, **y**, **priority** (1 = most urgent) and press **Add task**.
- **ALLOCATE** assigns pending tasks to the nearest free followers, highest priority first.
- **Clear** deletes all tasks; robots that were working rejoin the formation.
- The list shows each task's coordinates, priority, assigned agent and status
  (`PENDING`, `ASSIGNED`, `IN_PROGRESS`, `DONE`). When all tasks finish, the
  swarm regroups automatically. If there are more tasks than free followers,
  the rest wait and are assigned as followers become free.

**Communication log (real network events).** The most recent messages, e.g.
`F2->L HEARTBEAT LOST` or `L->F3 SCATTER DELAY=0.15s`. Red lines are lost
packets, orange are delayed. Because the position stream would flood the log,
it is summarised in one line ("`POSE_UPDATE stream (last 36): 34 ok / 2 lost`").
Entries come directly from the network model, so nothing here is simulated for
show.

**Command history.** Every command you issued, newest first, with the clock
time, success or `ERR`, and the engine's response.

**Events.** Short system notices with simulation time: `GOTO COMPLETE`,
`SCATTER COMPLETE`, `REGROUP COMPLETE`, `TASK A COMPLETE (Agent 1)`,
`ALL TASKS COMPLETE`, mode changes, forced-loss on/off, and any fault.

**Halt banner.** If the engine detects an invalid state (NaN/Inf), a red
banner covers the map with the failing agent and time, the simulation stops,
and further commands are refused until you press **Reset**.

---

## The Matplotlib window

Same engine, same drawing rules (fixed frame, persistent artists, live
stepping).

| Control | Action |
|---|---|
| Buttons | FOLLOW, SCATTER, REGROUP, STOP, RESUME, ALLOCATE, MODE (cycles AUTO → MANUAL → COMMAND) |
| Command box (bottom) | Type any command and press Enter. Typing here does not drive the robot. |
| Left-click on map | `GOTO` that point |
| Keys | W/S/A/D drive, SPACE stop, R regroup, X scatter, F follow |

Panels on the right show agents, network, task allocation, the real
communication log, command history and keyboard state. The title shows time,
mode, mission, formation error and minimum separation.

Matplotlib's own shortcuts for those keys (save, fullscreen, home, …) are
disabled while the view is open.

Extras: `view.save_frame("shot.png")`, `view.save_animation("run.gif", duration=15)`,
`view.run_steps(n)` for headless runs.

---

## Commands reference

Commands are case-insensitive and work from the dashboard command box, the
Matplotlib command box, or code via `sim.submit_command("...")`.

| Command | Effect |
|---|---|
| `GOTO x y` | Leader drives to world coordinate (x, y); followers keep formation. Out-of-world targets are clamped, NaN/inf rejected. |
| `MOVE dx dy` | Like GOTO, but relative to the leader's current position. |
| `SCATTER` | Followers disperse to individual safe points around the leader. |
| `REGROUP` | Followers return to their formation slots, then the mission becomes `FOLLOWING`. |
| `FOLLOW` | Restore normal leader–follower behaviour (cancels active tasks). |
| `STOP` | Stop everything safely and remember the mission. |
| `RESUME` | Continue the interrupted mission. |
| `FORMATION` | Show the current offsets. |
| `FORMATION WEDGE\|LINE\|COLUMN` | Change formation and regroup. |
| `STATUS` | One-screen summary of every agent, link state and network loss. |
| `TASK id x y [priority]` | Create a task (priority 1 = most urgent). |
| `ALLOCATE` | Assign pending tasks to free followers. |
| `TASKS` | List tasks. |
| `CLEARTASKS` | Delete all tasks. |
| `MODE AUTO\|MANUAL\|COMMAND` | Choose how the leader is driven. |

**Single-agent prefix.** `AGENT 2 GOTO 10 5`, `AGENT 3 STOP`, `AGENT 3 RESUME`.
Also `A2 …` and `F2 …`. `GOTO` works for the leader (`LEADER GOTO 5 5`, or no
prefix) and for any follower. `STOP` and `RESUME` work per **follower**; with no
prefix, `ALL`, or `LEADER` they act on the **whole swarm** (the leader cannot be
stopped on its own). Every other command acts on the whole swarm.

Comma separators are accepted: `GOTO 10,5`.

Any new motion command issued while stopped cancels the stop automatically.

---

## Keyboard control

| Key | Action |
|---|---|
| **W** / **↑** | Accelerate forward (hold) |
| **S** / **↓** | Accelerate in reverse (hold) |
| **A** / **←** | Turn left (hold) |
| **D** / **→** | Turn right (hold) |
| **SPACE** | `STOP` |
| **R** | `REGROUP` |
| **X** | `SCATTER` |
| **F** | `FOLLOW` |

Pressing a driving key switches to `MANUAL` mode automatically (and cancels a
running GOTO). Speed ramps up while a key is held and coasts back to zero when
released; turning ramps the same way. Keyboard input never moves robots
directly: it produces a desired speed and turn rate that goes through the same
boundary guard, actuator lag and physics as every other command.

In the browser, keys are ignored while you are typing in a text box or select.

---

## Modes and mission states

**Leader drive mode** (`MODE …`): how the leader is driven.

| Mode | Behaviour |
|---|---|
| `AUTO` | Follows a predefined smooth trajectory (the notebook's open-loop pattern). The boundary guard turns it away from walls. This is the default. |
| `MANUAL` | Driven by the keyboard. |
| `COMMAND` | Holds still until you issue a command such as `GOTO`. Issuing `GOTO` switches to this mode. |

**Mission state** (`MISSION …`): what the swarm as a whole is doing.

| State | Meaning |
|---|---|
| `FOLLOWING` | Followers hold formation slots. |
| `GOTO` | Leader is driving to a target. Returns to the previous state on arrival (`GOTO COMPLETE`). |
| `SCATTERING` | Followers are dispersing or holding scatter points. |
| `REGROUPING` | Followers are returning to slots. Becomes `FOLLOWING` on completion. |
| `TASK_EXECUTION` | Followers are working on allocated tasks. Auto-regroups when all tasks are done. |
| `STOPPED` | Everything stopped. `RESUME` restores the saved state. |

**Per-agent modes** shown in the Agents panel: `AUTO`, `MANUAL`, `IDLE` and
`GOTO` (leader or single follower), `FOLLOW`, `REGROUP`, `SCATTER`, `TASK`,
`STOPPED`.

---

## How it works

### One simulation tick (`MultiAgentSimulation.step`)

1. **Operator layer.** Keyboard one-shots run, then the command state machine advances (GOTO arrival, scatter/regroup completion, task progress).
2. **Leader.** Its command comes from `AUTO`, the keyboard, `GOTO` or idle. It passes through the boundary guard and a "yield to anyone ahead" rule, then through the RK4 plant (unicycle plus first-order actuator lag).
3. **Network.** The leader broadcasts `POSE_UPDATE`; unacknowledged commands are re-sent every 0.25 s; followers send a `HEARTBEAT` every 0.5 s; due packets are delivered and applied.
4. **Followers** (updated one after another, as in the original notebook). Each dead-reckons the leader from its last packet, picks the control law for its mode, adds avoidance and the boundary guard, and integrates.
5. **Safety and bookkeeping.** NaN/Inf check, hard clamp, formation error, minimum separation, link health, logging.

### Control

- **Formation tracking (FOLLOW / REGROUP):** slot = leader pose + rotated offset. Body-frame error drives a velocity PID and a heading PID. Gains adapt to the error size. Neighbour coupling keeps followers evenly spaced. A forward feed-forward uses the leader's actual speed.
- **Go-to-point (GOTO / SCATTER / TASK):** a heading controller plus speed proportional to remaining distance, so robots slow smoothly and stop inside a tolerance. Robots turn in place when the target is behind them.
- **Avoidance:** an artificial-potential field with a small tangential component (so head-on robots sidestep to one side), force capping, and heading bending, which prevents deadlocks.

### Network and dead reckoning

- Channel: two-state Gilbert–Elliott (good/bad) burst-loss model plus random latency of 1–4 steps (50–200 ms).
- Latency is **causal**: a packet arrives after its delay, not instantly.
- Dead reckoning: while packets are missing, followers extrapolate the leader's pose from its last position and velocity. In the upgraded engine the velocity decays exponentially (`dr_lambda`), so a long blackout settles into a stationary hold instead of drifting indefinitely.
- Link status: `OFFLINE` if nothing has arrived for 2 s; `DEGRADED` if recent loss is above 20 % or nothing has arrived for 0.6 s.
- Commands are acknowledged and retried (up to 60 attempts). The newest command to a follower replaces older unacknowledged ones.

### Safety layers (boundaries)

1. **Target validation.** Operator targets are checked; outside-world targets are clamped inside, NaN/inf rejected.
2. **Soft boundary guard.** Inside a margin zone (default 2 m) the command blends toward "face inward" and speed drops if moving outward. It fades out near a robot's own target so a goal near a wall can still be reached.
3. **Rigid formation shift.** If the formation would stick out of the world, the whole formation shifts inward as one block so slot spacing is preserved.
4. **Hard clamp.** Last-resort position clamp (counted; normally 0).

---

## Project structure

```
multi_agent_digital_twin/
├── README.md
├── requirements.txt
├── simulation/                 # the engine (no UI imports)
│   ├── config.py               # world bounds, Config dataclass, defaults
│   ├── kinematics.py           # RK4 unicycle plant, frame transforms
│   ├── agent.py                # Agent, AgentMode, LinkStatus, agent creation
│   ├── messages.py             # Message, CommEvent, MessageBus
│   ├── network.py              # Gilbert-Elliott channel, dead reckoning, Network
│   ├── collision.py            # avoidance, boundary guard, target validation, clamp
│   ├── formation.py            # presets, slots, scatter targets, regroup check
│   ├── controller.py           # PID, adaptive gains, follower and go-to-point laws
│   ├── tasks.py                # Task, Allocator, NearestAvailableAllocator, TaskManager
│   ├── commands.py             # parser, mission state machine, command handlers
│   ├── keyboard.py             # key state -> desired leader command
│   ├── logger.py               # CSV logging
│   ├── metrics.py              # ITAE/RMSE, Routh-Hurwitz, sweeps, Monte-Carlo
│   ├── simulation.py           # MultiAgentSimulation engine + simulate() wrapper
│   └── __init__.py
├── visualization/
│   ├── matplotlib_view.py      # interactive Matplotlib client
│   └── __init__.py
└── web/
    ├── app.py                  # FastAPI server + background simulation thread
    ├── templates/index.html    # dashboard page
    └── static/
        ├── app.js              # canvas rendering + controls
        └── style.css
```

Dependency rule: `simulation/` never imports `visualization/` or `web/`.
Both front ends talk to the engine only through its public API.

---

## Python API

```python
from simulation import MultiAgentSimulation, SimMode, Config

sim = MultiAgentSimulation(
    cfg=Config(),              # all tunables
    n_agents=5,                # 1 leader + 4 followers
    control_mode="advanced",   # "zoh" | "pidr" | "advanced"
    seed=42, loss_scale=1.0,   # network randomness / loss multiplier
    sim_mode=SimMode.AUTO,     # AUTO | MANUAL | COMMAND
    formation="WEDGE",         # WEDGE | LINE | COLUMN
)

sim.step()                          # advance one tick (dt = 0.05 s)
sim.run(30)                         # advance 30 seconds
res = sim.submit_command("SCATTER") # res.ok, res.message
sim.handle_key("w", True)           # keyboard event
state = sim.get_state()             # JSON-friendly snapshot (agents, targets, tasks,
                                    # network, messages, command history, ...)
sim.network.set_forced_loss(True)   # force 100 % packet loss
```

Useful extras: `sim.attach_logger(SimLogger())`, `sim.tasks`, `sim.bus.recent()`,
`sim.network.snapshot()`.

To plug in your own task-allocation strategy, subclass `Allocator`
(`simulation/tasks.py`) and pass it to `TaskManager`; the rest of the system
does not change.

---

## REST / WebSocket API

Start the server, then browse the interactive docs at
<http://127.0.0.1:8000/docs>.

| Method | Path | Body | Purpose |
|---|---|---|---|
| GET | `/` | | Dashboard page |
| GET | `/state` | | Full snapshot |
| GET | `/agents` | | Agent list |
| GET | `/messages?n=40&hide_pose=false` | | Communication log |
| GET | `/network` | | Network statistics |
| GET | `/tasks` | | Tasks and allocation text |
| POST | `/command` | `{"text": "GOTO 10 5"}` | Any text command |
| POST | `/goto` | `{"x":10,"y":5,"agent":"ALL"}` | `agent`: `ALL`, `LEADER`, or a number |
| POST | `/scatter` `/regroup` `/follow` | | Swarm commands |
| POST | `/stop` `/resume` | `{"agent":"ALL"}` (optional) | Stop / resume |
| POST | `/formation/{name}` | | `WEDGE`, `LINE`, `COLUMN` |
| POST | `/mode` | `{"mode":"AUTO"}` | Leader drive mode |
| POST | `/key` | `{"key":"w","down":true}` | Keyboard events |
| POST | `/tasks` | `{"id":"A","x":4,"y":7,"priority":1}` | Add a task |
| POST | `/tasks/allocate` `/tasks/clear` | | Allocate / clear |
| POST | `/network/loss` | `{"forced":true}` | Force 100 % packet loss |
| POST | `/sim/pause` | `{"paused":true}` | Pause the clock |
| POST | `/sim/speed` | `{"speed":2}` | 0.1 – 8 |
| POST | `/sim/reset` | `{"seed":42,"loss_scale":1,"control_mode":"advanced","sim_mode":"AUTO","n_agents":5,"formation":"WEDGE"}` | Restart |
| WS | `/ws` | | State stream at about 10 Hz |

Command endpoints return `{"ok": bool, "message": "...", "command": "..."}`.
Invalid request bodies return HTTP 422 with a validation message.

---

## Configuration

All tunables live in `simulation/config.py` (`Config`). Pass your own instance
to `MultiAgentSimulation(cfg=Config(...))`. Selected values:

| Setting | Default | Meaning |
|---|---|---|
| world bounds | x −4…22, y −7…9 | World rectangle in metres |
| `dt` | 0.05 s | Control and integration step (20 Hz) |
| `kp, ki, kd` | 1.8, 0.18, 0.65 | Follower PID gains |
| `max_v`, `max_w` | 1.25 m/s, 2.4 rad/s | Actuator limits |
| `boundary_margin` | 2.0 m | Width of the soft boundary zone |
| `target_margin` | 0.6 m | How far inside the wall targets are kept |
| `safe_distance` | 0.70 m | Avoidance radius |
| `yield_min_distance` | 0.45 m | Robots ahead closer than this block motion |
| `goto_tolerance` | 0.25 m | Leader arrival radius |
| `target_tolerance` | 0.30 m | Follower arrival radius (scatter / task) |
| `regroup_tolerance` | 0.35 m | Per-follower radius for `REGROUP COMPLETE` |
| `scatter_radius` | 3.0 m | Scatter ring radius |
| `packet_good_to_bad`, `packet_bad_to_good` | 0.025, 0.22 | Gilbert–Elliott transition probabilities |
| `base_delay_steps`, `max_delay_steps` | 1, 4 | Latency range (steps of `dt`) |
| `dr_lambda` | 0.12 /s | Dead-reckoning velocity decay |
| `heartbeat_period` | 0.5 s | Follower heartbeat interval |
| `cmd_retry_period` | 0.25 s | Command retry interval |
| `offline_timeout_s` | 2.0 s | Silence before a link is `OFFLINE` |

---

## Logging and metrics

**CSV log** (optional, off by default):

```python
from simulation import MultiAgentSimulation, SimLogger
sim = MultiAgentSimulation(logger=SimLogger("logs"))
sim.run(60)
sim.logger.close()
```

One row per agent per tick, in `logs/sim_<timestamp>.csv`:

`timestamp, agent_id, x, y, theta, v, omega, mode, target_x, target_y,
packet_sent, packet_received, packet_lost, packet_latency, task_id`

Packet columns count that tick's traffic: sent *by* the agent, and received or
lost *to* the agent.

**Metrics** (`simulation.metrics`): `compute_metrics` (ITAE, RMSE, peak error,
control effort, minimum separation, packet loss %, collision %),
`routh_hurwitz_cubic`, `stability_scan`, `robustness_sweep`, `monte_carlo`,
`summarize_monte_carlo`.

---

## Research mode (notebook compatibility)

`simulate()` runs a fixed-length batch and returns the same history dictionary
the original notebook produced:

```python
from simulation import simulate, compute_metrics, monte_carlo, summarize_monte_carlo

h = simulate("advanced", seed=101)          # "zoh" | "pidr" | "advanced"
print(compute_metrics(h))

results = monte_carlo(simulate, runs=30)
print(summarize_monte_carlo(results))
```

- `legacy=True` (default) reproduces the original notebook exactly: no
  boundary logic, instantaneous back-dated packets, constant feed-forward. The
  results match the notebook's to about 1e-9.
- `legacy=False` runs the upgraded engine (boundary guard, causal latency,
  decaying dead reckoning, avoidance improvements).

Note: in the baseline `zoh` and `pidr` modes there is deliberately no collision
avoidance (as in the notebook), so their minimum separation can be small. Only
`advanced` avoids collisions.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: simulation` | Run from the project root, e.g. `python -m web.app`. |
| Page says "reconnecting…" | The server stopped, or a proxy blocks WebSockets. The page falls back to polling and reconnects on its own. |
| A key seems stuck | Click an empty part of the page, or press and release the key. The page also releases all keys when the window loses focus. |
| Keys do nothing | You are focused in a text box or dropdown; click the map or the page background. |
| Command says `SIMULATION HALTED` | The engine detected an invalid state. Press **Reset**. |
| Followers are far from the leader with heavy loss | Expected. With long blackouts the leader estimate fades to a hold. Untick forced loss or lower **Loss x**. |
| Scatter targets show as grey x | The `SCATTER` command has not reached that follower yet; it is being retried. |
| Matplotlib window does not open | Use a desktop backend (for example `pip install pyqt5` or use TkAgg), or use the web dashboard. |
| Port already in use | `python -m web.app --port 8001`. |

---

## Known limitations

- A single simulation instance is shared by everyone connected to the server (open the dashboard in two browsers and both control the same swarm).
- `ALLOCATE` uses true follower positions, not positions reported by heartbeats.
- Followers only change mode when a command is actually delivered, so under heavy loss a command can take a moment to take effect (by design).
- There is no automated test-suite folder yet; the behaviour listed here was checked with scripted runs.
- The dashboard's canvas was tested for correct logic with a headless DOM, not yet visually in every browser.

---

## Roadmap

- `tests/` (kinematics, network, commands, tasks, boundaries, notebook regression) and `scenarios/` (default, scatter, regroup, tasks).
- Hungarian / auction task allocators (the `Allocator` interface is ready).
- Hardware-in-the-loop bridge (real ESP8266 pair behind the network interface).
- Webots: replace `kinematics.rk4_step` and the view; the commands, tasks, messages and control logic stay unchanged.
- Re-fit network parameters from measured ESP-NOW latency and loss on real hardware.