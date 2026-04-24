"""Data models for the S&OP War Room environment."""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    DEMAND = "demand_planner"
    SUPPLY = "supply_planner"
    FINANCE = "finance"
    CONSENSUS = "consensus_planner"
    SYSTEM = "system"


class Message(BaseModel):
    role: Role
    text: str
    turn: int


class WarRoomObservation(BaseModel):
    scenario_brief: str
    baseline_forecast: float
    transcript: List[Message]
    current_turn: int
    turns_remaining: int


class WarRoomAction(BaseModel):
    text: str


class StakeholderState(BaseModel):
    role: Role
    hidden_preference_pct: float
    preference_target: float


class EpisodeState(BaseModel):
    scenario_id: str
    baseline_forecast: float
    true_demand: float
    stakeholders: List[StakeholderState]
    transcript: List[Message] = Field(default_factory=list)
    current_turn: int = 1
    done: bool = False
    final_commit: Optional[float] = None
