"""Rule-based stakeholder simulators for the War Room."""
import random
from typing import List
from models import Role, Message


class BaseStakeholder:
    """Parent class. Each stakeholder has a hidden preference and emits messages."""
    role: Role = Role.SYSTEM

    def __init__(self, hidden_preference_pct: float, baseline: float, seed: int = 0):
        self.pref_pct = hidden_preference_pct
        self.baseline = baseline
        self.target = baseline * (1 + hidden_preference_pct / 100)
        self.rng = random.Random(seed)

    def opening(self, transcript: List[Message]) -> str:
        """Turn 1 message. Leaks preference indirectly."""
        raise NotImplementedError

    def reaction(self, transcript: List[Message], llm_probe: str) -> str:
        """Turn 3 message. Reacts to LLM's probe with sharper signal."""
        raise NotImplementedError


class DemandPlanner(BaseStakeholder):
    role = Role.DEMAND

    def opening(self, transcript):
        if self.pref_pct > 5:
            signals = ["strong pipeline momentum", "sell-through above plan",
                       "channel pull-forward in key accounts"]
        elif self.pref_pct < -5:
            signals = ["softening orders", "customer destocking signals",
                       "weaker sell-in vs last cycle"]
        else:
            signals = ["mixed signals across regions", "steady but unremarkable pipeline",
                       "no major surprises this cycle"]
        s = self.rng.choice(signals)
        return f"From a demand side, we're seeing {s}. I'd lean toward adjusting up from baseline."

    def reaction(self, transcript, llm_probe):
        # Extract a number from the probe if present; otherwise push harder
        if self.pref_pct > 5:
            return (f"That number feels conservative given what sales is reporting. "
                    f"We should be closer to {self.target:.0f} based on pipeline coverage.")
        elif self.pref_pct < -5:
            return (f"Actually the demand signal is weaker than I first said — "
                    f"I'd pull back toward {self.target:.0f}.")
        else:
            return f"I could live with something around {self.target:.0f}, no strong objection."


class SupplyPlanner(BaseStakeholder):
    role = Role.SUPPLY

    def opening(self, transcript):
        if self.pref_pct < -5:
            concerns = ["capacity is tight in Q+1", "supplier constraints on key components",
                        "we're already running hot on overtime"]
        elif self.pref_pct > 5:
            concerns = ["capacity is healthy, room to stretch",
                        "supplier base has opened up this cycle"]
        else:
            concerns = ["capacity roughly matches baseline", "no major supply flags"]
        c = self.rng.choice(concerns)
        demand_hint = ""
        if transcript:
            demand_hint = " Noted what demand said, but "
        return f"On supply,{demand_hint}{c}. We should be careful about over-committing."

    def reaction(self, transcript, llm_probe):
        if self.pref_pct < -5:
            return (f"That's too aggressive — we can realistically support {self.target:.0f} "
                    f"without emergency actions.")
        elif self.pref_pct > 5:
            return (f"Supply-wise we can actually go higher, up to {self.target:.0f} if needed.")
        else:
            return f"Supply can accommodate roughly {self.target:.0f}, within normal ops."


class Finance(BaseStakeholder):
    role = Role.FINANCE

    def opening(self, transcript):
        # Finance always anchors to budget and reacts to prior speakers
        prior = " ".join([m.text for m in transcript if m.role != self.role])
        tone = ""
        if "lean toward adjusting up" in prior or "stretch" in prior:
            tone = "I want to flag that we have a budget commitment already locked. "
        elif "pull back" in prior or "tight" in prior:
            tone = "Note that the budget assumes a specific revenue target. "
        return (f"{tone}From a finance perspective, we need to stay close to the committed "
                f"budget number — roughly {self.target:.0f}.")

    def reaction(self, transcript, llm_probe):
        return (f"Any deviation from {self.target:.0f} needs a variance explanation "
                f"to the CFO. I'd strongly prefer we anchor there.")
