"""Local HTTP client for smoke-testing the WarRoom env."""
import requests


class WarRoomClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, scenario_id: str = None, seed: int = 0):
        r = requests.post(f"{self.base_url}/reset",
                          json={"scenario_id": scenario_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, text: str):
        r = requests.post(f"{self.base_url}/step",
                          json={"action": {"text": text}})
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    c = WarRoomClient()
    obs = c.reset(scenario_id="medium_split", seed=42)
    print(f"Turn {obs['current_turn']}, {len(obs['transcript'])} messages in transcript")

    r1 = c.step("What if we aim for 1580?")
    print(f"After probe: turn={r1['observation']['current_turn']}, done={r1['done']}")

    r2 = c.step("Balancing all inputs, FINAL FORECAST: 1550")
    print(f"After commit: done={r2['done']}, reward={r2['reward']:.3f}")
    print(f"Info: {r2['info']}")
