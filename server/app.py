"""FastAPI app exposing the WarRoom environment over HTTP."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from models import WarRoomAction, WarRoomObservation
from server.environment import WarRoomEnvironment
from server.grader import grade_episode


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
    app = FastAPI(title="S&OP War Room Environment")
    env = WarRoomEnvironment()

    @app.get("/")
    def root():
        return {"name": "sop-war-room", "status": "ok"}

    @app.post("/reset", response_model=WarRoomObservation)
    def reset(req: ResetRequest):
        return env.reset(scenario_id=req.scenario_id, seed=req.seed)

    @app.post("/step", response_model=StepResponse)
    def step(req: StepRequest):
        obs, _reward, done, info = env.step(req.action)
        # Grade only on episode end
        reward = 0.0
        if done:
            reward = grade_episode(env.state)
            info["reward_breakdown"] = info.get("reward_breakdown", {})
        return StepResponse(observation=obs, reward=reward, done=done, info=info)

    return app


app = create_app()
