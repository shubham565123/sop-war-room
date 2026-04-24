"""Episode engine for the S&OP War Room."""
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Tuple
from models import (
    Role, Message, WarRoomObservation, WarRoomAction,
    StakeholderState, EpisodeState
)
from stakeholders import DemandPlanner, SupplyPlanner, Finance


SCENARIOS = [
    {
        "id": "easy_aligned",
        "brief": "Q3 forecast for a mature detergent SKU in steady market.",
        "baseline": 1000.0,
        "prefs": {"demand": +5, "supply": -3, "finance": -2},
        "true_demand_noise": 30,
    },
    {
        "id": "medium_split",
        "brief": "Q4 forecast for a seasonal beverage, mixed signals.",
        "baseline": 1500.0,
        "prefs": {"demand": +12, "supply": -7, "finance": -2},
        "true_demand_noise": 80,
    },
    {
        "id": "hard_divergent",
        "brief": "H1 forecast for a new skincare launch, high uncertainty.",
        "baseline": 800.0,
        "prefs": {"demand": +20, "supply": -10, "finance": -2},
        "true_demand_noise": 120,
    },
]


class WarRoomEnvironment:
    """Runs one episode of the S&OP War Room."""

    def __init__(self):
        self.state: Optional[EpisodeState] = None
        self._demand: Optional[DemandPlanner] = None
        self._supply: Optional[SupplyPlanner] = None
        self._finance: Optional[Finance] = None
        self._rng = random.Random()

    def reset(self, scenario_id: Optional[str] = None, seed: int = 0) -> WarRoomObservation:
        """Start a new episode. Returns the first observation the LLM sees (turn 2)."""
        self._rng = random.Random(seed)
        scenario = self._pick_scenario(scenario_id)
        baseline = scenario["baseline"]

        self._demand = DemandPlanner(scenario["prefs"]["demand"], baseline, seed)
        self._supply = SupplyPlanner(scenario["prefs"]["supply"], baseline, seed)
        self._finance = Finance(scenario["prefs"]["finance"], baseline, seed)

        stakeholders = [
            StakeholderState(role=Role.DEMAND,
                             hidden_preference_pct=scenario["prefs"]["demand"],
                             preference_target=self._demand.target),
            StakeholderState(role=Role.SUPPLY,
                             hidden_preference_pct=scenario["prefs"]["supply"],
                             preference_target=self._supply.target),
            StakeholderState(role=Role.FINANCE,
                             hidden_preference_pct=scenario["prefs"]["finance"],
                             preference_target=self._finance.target),
        ]

        # Ground truth: biased toward average of prefs + noise
        avg_pref = sum(scenario["prefs"].values()) / 3
        true_demand = baseline * (1 + avg_pref / 100) + self._rng.uniform(
            -scenario["true_demand_noise"], scenario["true_demand_noise"]
        )

        self.state = EpisodeState(
            scenario_id=scenario["id"],
            baseline_forecast=baseline,
            true_demand=true_demand,
            stakeholders=stakeholders,
        )

        # Run Turn 1: all three stakeholders speak in sequence
        self._run_opening_round()

        # Advance to turn 2 — the LLM's first speaking turn
        self.state.current_turn = 2
        return self._build_observation()

    def step(self, action: WarRoomAction) -> Tuple[WarRoomObservation, float, bool, dict]:
        """Apply one LLM action. Returns (obs, reward, done, info)."""
        if self.state is None or self.state.done:
            raise RuntimeError("Episode not active. Call reset() first.")

        # Record the LLM's message
        self.state.transcript.append(
            Message(role=Role.CONSENSUS, text=action.text, turn=self.state.current_turn)
        )

        if self.state.current_turn == 2:
            # Run Turn 3: reaction round
            self._run_reaction_round(llm_probe=action.text)
            self.state.current_turn = 4
            return self._build_observation(), 0.0, False, {}

        elif self.state.current_turn == 4:
            # Final commit — grading happens elsewhere in grader.py
            self.state.final_commit = self._extract_forecast(action.text)
            self.state.done = True
            return self._build_observation(), 0.0, True, {
                "final_commit": self.state.final_commit,
                "true_demand": self.state.true_demand,
            }

        raise RuntimeError(f"Unexpected turn: {self.state.current_turn}")

    # --- internals ---

    def _pick_scenario(self, scenario_id: Optional[str]) -> dict:
        if scenario_id:
            for s in SCENARIOS:
                if s["id"] == scenario_id:
                    return s
            raise ValueError(f"Unknown scenario_id: {scenario_id}")
        return self._rng.choice(SCENARIOS)

    def _run_opening_round(self):
        """Turn 1: demand, supply, finance each speak once, in order."""
        for sh in [self._demand, self._supply, self._finance]:
            msg = sh.opening(self.state.transcript)
            self.state.transcript.append(Message(role=sh.role, text=msg, turn=1))

    def _run_reaction_round(self, llm_probe: str):
        """Turn 3: stakeholders react to the LLM's turn-2 probe."""
        for sh in [self._demand, self._supply, self._finance]:
            msg = sh.reaction(self.state.transcript, llm_probe)
            self.state.transcript.append(Message(role=sh.role, text=msg, turn=3))

    def _build_observation(self) -> WarRoomObservation:
        turns_remaining = 4 - self.state.current_turn  # 2 on turn 2, 0 on turn 4
        return WarRoomObservation(
            scenario_brief=self._scenario_brief(),
            baseline_forecast=self.state.baseline_forecast,
            transcript=list(self.state.transcript),
            current_turn=self.state.current_turn,
            turns_remaining=turns_remaining,
        )

    def _scenario_brief(self) -> str:
        for s in SCENARIOS:
            if s["id"] == self.state.scenario_id:
                return s["brief"]
        return ""

    @staticmethod
    def _extract_forecast(text: str) -> Optional[float]:
        """Pull the number from 'FINAL FORECAST: X' if present."""
        import re
        m = re.search(r"FINAL FORECAST\s*:\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
        return None
