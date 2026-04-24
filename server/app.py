"""FastAPI app exposing the WarRoom environment over HTTP.

Implements the OpenEnv HTTP contract v0.1.0:
  GET  /            — simple root
  GET  /health      — liveness probe
  GET  /metadata    — env name, description, version
  GET  /schema      — action, observation, state JSON schemas
  POST /mcp         — MCP JSON-RPC bridge (minimal stub)
  POST /reset       — start an episode
  POST /step        — advance the episode
  GET  /state       — current episode state snapshot
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Any, Dict
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

from models import (
    WarRoomAction, WarRoomObservation, EpisodeState
)
from server.environment import WarRoomEnvironment
from server.grader import grade_episode, compute_breakdown


VERSION = "0.1.0"
ENV_NAME = "sop-war-room"
ENV_DESCRIPTION = (
    "Multi-agent S&OP (Sales & Operations Planning) negotiation environment. "
    "A Consensus Planner LLM negotiates with 3 rule-based stakeholder agents "
    "(Demand, Supply, Finance) with hidden competing preferences over 4 turns, "
    "then commits to a balanced forecast. Rewarded on format, accuracy, "
    "consensus quality, information extraction, and negotiation efficiency."
)


class ResetRequest(BaseModel):
    scenario_id: Optional[str] = None
    seed: int = 0


class StepRequest(BaseModel):
    action: WarRoomAction


class StepResponse(BaseModel):
    observation: WarRoomObservation
    reward: float
    done: bool
    info: dict


def create_app() -> FastAPI:
    app = FastAPI(title="S&OP War Room Environment", version=VERSION)
    env = WarRoomEnvironment()

    # --- OpenEnv contract: introspection ---

    @app.get("/")
    def root():
        return {"name": ENV_NAME, "status": "ok", "version": VERSION}

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/metadata")
    def metadata():
        return {
            "name": ENV_NAME,
            "description": ENV_DESCRIPTION,
            "version": VERSION,
            "author": "shubhamyeole565",
            "license": "Apache-2.0",
            "tasks": ["task_easy", "task_medium", "task_hard"],
        }

    @app.get("/schema")
    def schema():
        return {
            "action": WarRoomAction.model_json_schema(),
            "observation": WarRoomObservation.model_json_schema(),
            "state": EpisodeState.model_json_schema(),
        }

    # --- Minimal MCP (JSON-RPC) stub ---

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        """Minimal JSON-RPC handler exposing env capabilities over MCP."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        method = body.get("method", "")
        rpc_id = body.get("id", 1)

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": ENV_NAME, "version": VERSION},
                },
            }
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "tools": [
                        {"name": "reset", "description": "Start an episode"},
                        {"name": "step", "description": "Advance the episode"},
                        {"name": "state", "description": "Inspect current state"},
                    ]
                },
            }
        # Default: acknowledge
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"ack": True, "method": method},
        }

    # --- OpenEnv contract: gym-style ---

    @app.post("/reset", response_model=WarRoomObservation)
    def reset(req: ResetRequest):
        return env.reset(scenario_id=req.scenario_id, seed=req.seed)

    @app.post("/step", response_model=StepResponse)
    def step(req: StepRequest):
        obs, _reward, done, info = env.step(req.action)
        reward = 0.0
        if done:
            reward = grade_episode(env.state)
            info["reward_breakdown"] = compute_breakdown(env.state)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)

    @app.get("/state")
    def state() -> Dict[str, Any]:
        """Return current episode state, or a default snapshot if not started."""
        if env.state is None:
            return {
                "active": False,
                "scenario_id": None,
                "current_turn": 0,
                "done": False,
            }
        return {
            "active": True,
            "scenario_id": env.state.scenario_id,
            "current_turn": env.state.current_turn,
            "done": env.state.done,
            "transcript_length": len(env.state.transcript),
            "final_commit": env.state.final_commit,
        }

    return app


app = create_app()


def main():
    """Entry point for `server` script and direct execution."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
