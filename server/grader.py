"""Episode grader. Full rubric implementation in Block 3."""
from models import EpisodeState


def grade_episode(state: EpisodeState) -> float:
    """Placeholder — returns a crude accuracy signal for smoke testing."""
    if state.final_commit is None:
        return 0.01
    # Simple normalized error, clamped
    err = abs(state.final_commit - state.true_demand) / state.true_demand
    score = max(0.01, min(0.99, 1.0 - err))
    return score
