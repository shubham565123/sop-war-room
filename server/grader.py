"""Episode grader with 5 rubrics for the S&OP War Room."""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
from models import EpisodeState, Role


# --- Rubric 1: Format Gate ---

def format_gate(state: EpisodeState) -> float:
    """Hard constraint: did turn 4 contain a valid FINAL FORECAST: X?"""
    if state.final_commit is None:
        return 0.0
    if state.final_commit <= 0:
        return 0.0
    return 1.0


# --- Rubric 2: Accuracy (tightened) ---

def accuracy_score(state: EpisodeState) -> float:
    """Close to true_demand. Full credit within 1%, zero beyond 8%."""
    if state.final_commit is None:
        return 0.0
    err = abs(state.final_commit - state.true_demand)
    normalized_err = err / state.baseline_forecast
    # Tightened: full credit <=1% off, zero >=8% off
    if normalized_err <= 0.01:
        return 1.0
    if normalized_err >= 0.08:
        return 0.0
    # Linear between 1% and 8%
    return 1.0 - (normalized_err - 0.01) / 0.07


# --- Rubric 3: Consensus (tightened) ---

def consensus_score(state: EpisodeState) -> float:
    """Close to balanced compromise of stakeholder targets. Full credit <=1%, zero >=6%."""
    if state.final_commit is None:
        return 0.0
    targets = [sh.preference_target for sh in state.stakeholders]
    ideal_consensus = sum(targets) / len(targets)
    err = abs(state.final_commit - ideal_consensus)
    normalized_err = err / state.baseline_forecast
    if normalized_err <= 0.01:
        return 1.0
    if normalized_err >= 0.06:
        return 0.0
    return 1.0 - (normalized_err - 0.01) / 0.05


# --- Rubric 4: Extraction ---

def extraction_score(state: EpisodeState) -> float:
    """Did the final commit reference stakeholder targets (within 5% tolerance)?"""
    turn4_msgs = [m.text for m in state.transcript
                  if m.role == Role.CONSENSUS and m.turn == 4]
    if not turn4_msgs:
        return 0.0
    final_text = turn4_msgs[0].lower()
    nums_in_text = [float(n.replace(",", "")) for n in
                    re.findall(r"\b(\d{2,5}(?:,\d{3})*(?:\.\d+)?)\b", final_text)]
    if not nums_in_text:
        return 0.0
    hits = 0
    for sh in state.stakeholders:
        target = sh.preference_target
        if any(abs(n - target) / target <= 0.05 for n in nums_in_text):
            hits += 1
    return hits / 3.0


# --- Rubric 5: Efficiency ---

def efficiency_score(state: EpisodeState) -> float:
    """Turn 2: active probing (question + number best, either alone partial)."""
    turn2_msgs = [m.text for m in state.transcript
                  if m.role == Role.CONSENSUS and m.turn == 2]
    if not turn2_msgs:
        return 0.0
    probe = turn2_msgs[0]
    has_question = "?" in probe
    has_number = bool(re.search(r"\b\d{2,5}\b", probe))
    if has_question and has_number:
        return 1.0
    if has_question or has_number:
        return 0.6
    return 0.2


# --- Composition ---

WEIGHTS = {
    "accuracy": 0.40,
    "consensus": 0.30,
    "extraction": 0.20,
    "efficiency": 0.10,
}


def grade_episode(state: EpisodeState) -> float:
    breakdown = compute_breakdown(state)
    gate = breakdown["format"]
    weighted = (WEIGHTS["accuracy"] * breakdown["accuracy"]
                + WEIGHTS["consensus"] * breakdown["consensus"]
                + WEIGHTS["extraction"] * breakdown["extraction"]
                + WEIGHTS["efficiency"] * breakdown["efficiency"])
    final = gate * weighted
    return max(0.01, min(0.99, final))


def compute_breakdown(state: EpisodeState) -> Dict[str, float]:
    """Return per-rubric scores. Sub-rubrics zeroed out when format gate fails."""
    gate = format_gate(state)
    if gate == 0.0:
        return {"format": 0.0, "accuracy": 0.0, "consensus": 0.0,
                "extraction": 0.0, "efficiency": 0.0}
    return {
        "format": gate,
        "accuracy": accuracy_score(state),
        "consensus": consensus_score(state),
        "extraction": extraction_score(state),
        "efficiency": efficiency_score(state),
    }
