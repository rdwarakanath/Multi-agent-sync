"""
app.py -- FastAPI dashboard server for the multi-agent digital twin.

Architecture
------------
    browser  --HTTP (commands)-->  FastAPI  --public API-->  MultiAgentSimulation
    browser  <--WebSocket / GET /state--  FastAPI  <--get_state()--  (same engine)

* ONE engine instance runs in a background thread at real time
  (`dt / speed` seconds per tick).  All access goes through `runner.lock`.
* The UI never touches agent state.  Every button, click and key becomes a
  command / key event that the engine validates and applies.
* The engine is the same object the Matplotlib view and (later) Webots use.

Run (from the project root):

    python -m web.app                       # http://127.0.0.1:8000
    python -m web.app --host 0.0.0.0 --port 8080
    uvicorn web.app:app --reload

Endpoints
---------
GET  /                  dashboard page
GET  /state             full snapshot            GET /agents      agent list
GET  /messages          communication log        GET /network     network statistics
GET  /tasks             task list + allocation
POST /command           {"text": "GOTO 10 5"}    any text command
POST /goto              {"x":10,"y":5,"agent":"ALL|LEADER|2"}
POST /scatter /regroup /follow
POST /stop /resume      {"agent":"ALL|LEADER|2"} (optional body)
POST /formation/{name}  WEDGE | LINE | COLUMN
POST /mode              {"mode":"AUTO|MANUAL|COMMAND"}
POST /key               {"key":"w","down":true}  keyboard events
POST /tasks             {"id":"A","x":4,"y":7,"priority":1}
POST /tasks/allocate    POST /tasks/clear
POST /network/loss      {"forced":true}          force 100 % packet loss (test)
POST /sim/pause         {"paused":true}          POST /sim/speed {"speed":2}
POST /sim/reset         {"seed":42,"loss_scale":1,"control_mode":"advanced",...}
WS   /ws                state stream (~10 Hz)
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from simulation import MultiAgentSimulation, SimMode
from simulation.config import Config
from simulation.formation import FORMATION_PRESETS

BASE_DIR = Path(__file__).resolve().parent
STATE_HZ = 10.0                      # WebSocket push rate


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
class CommandBody(BaseModel):
    text: str = Field(..., max_length=200)


class GotoBody(BaseModel):
    x: float
    y: float
    agent: Optional[str] = "ALL"


class AgentBody(BaseModel):
    agent: Optional[str] = "ALL"


class ModeBody(BaseModel):
    mode: str


class KeyBody(BaseModel):
    key: str = Field(..., max_length=16)
    down: bool = True


class TaskBody(BaseModel):
    id: str = Field(..., min_length=1, max_length=16)
    x: float
    y: float
    priority: int = 1


class LossBody(BaseModel):
    forced: bool


class PauseBody(BaseModel):
    paused: bool


class SpeedBody(BaseModel):
    speed: float = Field(..., ge=0.1, le=8.0)


class ResetBody(BaseModel):
    seed: int = 42
    loss_scale: float = Field(1.0, ge=0.0, le=10.0)
    control_mode: str = "advanced"
    sim_mode: str = "AUTO"
    n_agents: int = Field(5, ge=2, le=12)
    formation: str = "WEDGE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean(obj: Any) -> Any:
    """Make a structure JSON-safe: NaN / inf -> None (Starlette rejects them)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _prefix(agent: Optional[str]) -> str:
    """Translate the agent selector into the command parser's prefix."""
    if agent is None or str(agent).upper() == "ALL":
        return ""
    a = str(agent).strip().upper()
    if a == "LEADER":
        return "LEADER "
    if a.isdigit():
        return f"AGENT {int(a)} "
    return ""


class SimulationRunner:
    """Owns the engine and the real-time stepping thread."""

    def __init__(self, **sim_kwargs):
        self._kwargs = dict(sim_kwargs)
        self.lock = threading.RLock()
        self.sim = MultiAgentSimulation(**self._kwargs)
        self.paused = False
        self.speed = 1.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="sim-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self.lock:
                if not self.paused:
                    try:
                        self.sim.step()
                    except Exception as exc:                 # never kill the thread
                        self.sim.halted = True
                        self.sim.fault = f"ENGINE ERROR: {type(exc).__name__}: {exc}"
                period = self.sim.dt / self.speed
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:                                            # fell behind: do not spiral
                next_t = time.perf_counter()

    # ---- access ---------------------------------------------------------
    def snapshot(self, n_messages: int = 40) -> dict:
        with self.lock:
            st = self.sim.get_state(n_messages)
            st["runner"] = {"paused": self.paused, "speed": self.speed}
        return _clean(st)

    def command(self, text: str) -> dict:
        with self.lock:
            res = self.sim.submit_command(text)
        return {"ok": res.ok, "message": res.message, "command": res.command}

    def reset(self, body: ResetBody) -> dict:
        try:
            mode = SimMode(body.sim_mode.upper())
            if body.control_mode not in ("zoh", "pidr", "advanced"):
                raise ValueError("control_mode must be zoh, pidr or advanced")
            if body.formation.upper() not in FORMATION_PRESETS:
                raise ValueError(f"formation must be one of {FORMATION_PRESETS}")
        except ValueError as exc:
            return {"ok": False, "message": f"RESET REJECTED: {exc}"}
        kwargs = dict(self._kwargs)
        kwargs.update(seed=body.seed, loss_scale=body.loss_scale,
                      control_mode=body.control_mode, sim_mode=mode,
                      n_agents=body.n_agents, formation=body.formation.upper())
        with self.lock:
            self.sim = MultiAgentSimulation(**kwargs)
            self.paused = False
        return {"ok": True, "message": "SIMULATION RESET"}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(**sim_kwargs) -> FastAPI:
    runner = SimulationRunner(**sim_kwargs)
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        runner.start()
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(title="Multi-Agent Digital Twin", lifespan=lifespan)
    app.state.runner = runner
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    # ---- pages ---------------------------------------------------------
    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html",
                                          {"title": "Multi-Agent Digital Twin"})

    # ---- read endpoints --------------------------------------------------
    @app.get("/state")
    def state():
        return runner.snapshot()

    @app.get("/agents")
    def agents():
        return runner.snapshot()["agents"]

    @app.get("/messages")
    def messages(n: int = 40, hide_pose: bool = False):
        with runner.lock:
            return _clean(runner.sim.bus.recent(max(1, min(n, 300)), hide_pose))

    @app.get("/network")
    def network():
        with runner.lock:
            return _clean(runner.sim.network.snapshot())

    @app.get("/tasks")
    def tasks():
        with runner.lock:
            return _clean({"tasks": runner.sim.tasks.to_list(),
                           "allocation": runner.sim.tasks.summary_lines()})

    # ---- commands ----------------------------------------------------------
    @app.post("/command")
    def command(body: CommandBody):
        return runner.command(body.text)

    @app.post("/goto")
    def goto(body: GotoBody):
        return runner.command(f"{_prefix(body.agent)}GOTO {body.x!r} {body.y!r}")

    @app.post("/scatter")
    def scatter():
        return runner.command("SCATTER")

    @app.post("/regroup")
    def regroup():
        return runner.command("REGROUP")

    @app.post("/follow")
    def follow():
        return runner.command("FOLLOW")

    @app.post("/stop")
    def stop(body: AgentBody | None = None):
        return runner.command(f"{_prefix(body.agent if body else None)}STOP")

    @app.post("/resume")
    def resume(body: AgentBody | None = None):
        return runner.command(f"{_prefix(body.agent if body else None)}RESUME")

    @app.post("/formation/{name}")
    def formation(name: str):
        return runner.command(f"FORMATION {name}")

    @app.post("/mode")
    def mode(body: ModeBody):
        return runner.command(f"MODE {body.mode}")

    # ---- keyboard ------------------------------------------------------------
    @app.post("/key")
    def key(body: KeyBody):
        with runner.lock:
            k = runner.sim.handle_key(body.key, body.down)
        return {"ok": k is not None, "key": k}

    # ---- tasks ------------------------------------------------------------------
    @app.post("/tasks")
    def add_task(body: TaskBody):
        return runner.command(f"TASK {body.id} {body.x!r} {body.y!r} {body.priority}")

    @app.post("/tasks/allocate")
    def allocate():
        return runner.command("ALLOCATE")

    @app.post("/tasks/clear")
    def clear_tasks():
        return runner.command("CLEARTASKS")

    # ---- experiment hooks ----------------------------------------------------------
    @app.post("/network/loss")
    def forced_loss(body: LossBody):
        with runner.lock:
            runner.sim.network.set_forced_loss(body.forced)
            runner.sim.log_status("FORCED 100% PACKET LOSS " + ("ON" if body.forced else "OFF"))
        return {"ok": True, "forced": body.forced}

    @app.post("/sim/pause")
    def pause(body: PauseBody):
        runner.paused = body.paused
        return {"ok": True, "paused": runner.paused}

    @app.post("/sim/speed")
    def speed(body: SpeedBody):
        runner.speed = body.speed
        return {"ok": True, "speed": runner.speed}

    @app.post("/sim/reset")
    def reset(body: ResetBody):
        return runner.reset(body)

    # ---- live stream ------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                payload = await run_in_threadpool(runner.snapshot)
                await websocket.send_json(payload)
                await asyncio.sleep(1.0 / STATE_HZ)
        except (WebSocketDisconnect, RuntimeError):
            pass

    # ---- uniform JSON for unexpected errors (UI must never see an HTML 500) -------------
    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):
        return JSONResponse({"ok": False, "message": f"SERVER ERROR: {type(exc).__name__}: {exc}"},
                            status_code=500)

    return app


app = create_app()


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="Multi-Agent Digital Twin dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    uvicorn.run(create_app(seed=args.seed), host=args.host, port=args.port)


if __name__ == "__main__":
    main()