"""
Inference script for S&OP War Room environment.

Drives the environment over HTTP through 4 turns:
  Turn 1: 3 stakeholder openers (auto, server-side)
  Turn 2: LLM probes
  Turn 3: 3 stakeholder reactions (auto, server-side)
  Turn 4: LLM commits with 'FINAL FORECAST: X'

Env vars (per hackathon Sample Reference Script):
    API_BASE_URL   LLM endpoint (default: HF router)
    MODEL_NAME     Model to call
    HF_TOKEN       Auth token (or API_KEY)
    ENV_BASE_URL   Running War Room env (default: http://localhost:8000)

Runs three tasks (easy_aligned, medium_split, hard_divergent) and prints
one [START]...[END] block per task for the validator to parse.
"""
import os
import re
import sys
import textwrap
from typing import List, Dict, Optional

import requests
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")

TEMPERATURE = 0.2
MAX_TOKENS = 200
SCENARIOS = ["easy_aligned", "medium_split", "hard_divergent"]
SEEDS = [int(s) for s in os.getenv("SEEDS", "42").split(",")]


SYSTEM_PROMPT = textwrap.dedent("""
You are the Consensus Planner in an S&OP (Sales & Operations Planning) meeting.
Three stakeholders are present:
  - Demand Planner (market-facing, often pushes higher)
  - Supply Planner (capacity-facing, often pushes lower)
  - Finance (budget-anchored, conservative)

Each stakeholder has a hidden target forecast. Your job is to:
  1. On your first turn: ask a clarifying question OR float a preliminary number to
     provoke sharper signals from them.
  2. On your final turn: commit to a forecast that balances all three positions
     while being grounded in the likely true demand.

On your final turn, your message MUST end with the exact line:
    FINAL FORECAST: <number>

Reference each stakeholder's target number in your final justification when possible.
Be concise. One short paragraph per turn.
""").strip()


def build_user_prompt(observation: Dict) -> str:
    """Render the current observation as a user message for the LLM."""
    transcript_lines = []
    for m in observation["transcript"]:
        role = m["role"].replace("_", " ").title()
        transcript_lines.append(f"[{role}]: {m['text']}")
    transcript_str = "\n".join(transcript_lines) if transcript_lines else "(empty)"

    turn = observation["current_turn"]
    turn_instr = ("This is your PROBE turn. Ask a clarifying question or "
                  "propose a preliminary number to elicit sharper stakeholder signals. "
                  "Do NOT commit yet.") if turn == 2 else \
                 ("This is your FINAL COMMIT turn. End your message with "
                  "'FINAL FORECAST: <number>'. Reference each stakeholder's target.")

    return textwrap.dedent(f"""
        Scenario: {observation['scenario_brief']}
        Baseline forecast: {observation['baseline_forecast']}
        Current turn: {turn} (of 4)

        --- Meeting transcript so far ---
        {transcript_str}
        --- End transcript ---

        {turn_instr}
    """).strip()


def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Single OpenAI-compatible chat completion."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}. Using fallback.", file=sys.stderr)
        return fallback_response(user_prompt)


def fallback_response(user_prompt: str) -> str:
    """If LLM fails, return a minimally-valid response so the env can complete."""
    if "FINAL COMMIT" in user_prompt:
        # Extract baseline to produce a valid commit
        m = re.search(r"Baseline forecast:\s*([\d.]+)", user_prompt)
        baseline = float(m.group(1)) if m else 1000.0
        return f"Acknowledging all positions. FINAL FORECAST: {baseline:.0f}"
    return "Could we align around the baseline number?"


def run_episode(client: OpenAI, scenario_id: str, seed: int = 42) -> Dict:
    """Run one full 4-turn episode against the HTTP env. Returns final info dict."""
    # Reset
    r = requests.post(f"{ENV_BASE_URL}/reset",
                      json={"scenario_id": scenario_id, "seed": seed},
                      timeout=30)
    r.raise_for_status()
    obs = r.json()

    # Turn 2: LLM probes
    probe_prompt = build_user_prompt(obs)
    probe = call_llm(client, SYSTEM_PROMPT, probe_prompt)
    r = requests.post(f"{ENV_BASE_URL}/step",
                      json={"action": {"text": probe}},
                      timeout=30)
    r.raise_for_status()
    step1 = r.json()

    # Turn 4: LLM commits
    commit_prompt = build_user_prompt(step1["observation"])
    commit = call_llm(client, SYSTEM_PROMPT, commit_prompt)
    r = requests.post(f"{ENV_BASE_URL}/step",
                      json={"action": {"text": commit}},
                      timeout=30)
    r.raise_for_status()
    step2 = r.json()

    return {
        "scenario_id": scenario_id,
        "probe": probe,
        "commit": commit,
        "reward": step2["reward"],
        "info": step2["info"],
    }


def main() -> None:
    if not API_KEY:
        print("[WARN] No HF_TOKEN/API_KEY set. LLM calls will use fallback.",
              file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    # Quick env health check
    try:
        requests.get(f"{ENV_BASE_URL}/", timeout=5).raise_for_status()
    except Exception as exc:
        print(f"[ERROR] Env not reachable at {ENV_BASE_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    import json, statistics
    all_results = []
    n_total = len(SCENARIOS) * len(SEEDS)
    n_done = 0
    for scenario_id in SCENARIOS:
        for seed in SEEDS:
            n_done += 1
            print(f"[{n_done}/{n_total}] {scenario_id} seed={seed}...", file=sys.stderr)
            result = run_episode(client, scenario_id, seed=seed)
            all_results.append({**result, "seed": seed})
            # Emit [START]...[END] only for the first seed of each task,
            # preserving the validator contract regardless of SEEDS setting.
            if seed == SEEDS[0]:
                print(f"[START]")
                print(f"task: {scenario_id}")
                print(f"probe: {result['probe']}")
                print(f"commit: {result['commit']}")
                print(f"reward: {result['reward']:.4f}")
                print(f"breakdown: {result['info'].get('reward_breakdown', {})}")
                print(f"[END]")
                print()

    # Multi-seed summary (only when SEEDS has more than one entry)
    if len(SEEDS) > 1:
        print("\n=== Multi-seed baseline summary ===", file=sys.stderr)
        by_task = {}
        for r in all_results:
            by_task.setdefault(r['scenario_id'], []).append(r['reward'])
        for task, rewards in by_task.items():
            mean = statistics.mean(rewards)
            std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
            print(f"  {task}: mean={mean:.4f} std={std:.4f} n={len(rewards)}", file=sys.stderr)
        all_rewards = [r['reward'] for r in all_results]
        print(f"  OVERALL: mean={statistics.mean(all_rewards):.4f} std={statistics.stdev(all_rewards):.4f}", file=sys.stderr)
        with open("baseline_results.json", "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "seeds": SEEDS,
                "scenarios": SCENARIOS,
                "results": all_results,
            }, f, indent=2, default=str)
        print(f"  -> saved {len(all_results)} episodes to baseline_results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
