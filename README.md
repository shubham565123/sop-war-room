---
title: S&OP War Room
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# S&OP War Room

**A multi-agent Sales & Operations Planning negotiation environment for OpenEnv.**

Built for the Meta × PyTorch × Hugging Face OpenEnv Hackathon India — Grand Finale, April 25–26, 2026.
Addresses **Theme 1: Multi-Agent Interactions** — single-agent RL in a multi-agent environment.

---

## Links & Artifacts

> This section is progressively filled in across the onsite hackathon. If a link is a placeholder, that artifact is in active development.

| Artifact | Link |
|---|---|
| Hugging Face Space (live environment) | _to be added after deploy_ |
| GitHub repository | https://github.com/shubham565123/sop-war-room |
| Training notebook (Colab, TRL + Unsloth + Qwen2.5-3B GRPO) | _to be added after Day 1 training run_ |
| Reward / loss plots | `plots/` (to be committed after training) |
| Blog post (HF Hub or in-repo) | _to be added (see `blog/post.md`)_ |
| Demo video (<2 min, YouTube) | _optional — to be added if produced_ |

---

## What it is

A Consensus Planner LLM sits at a simulated S&OP meeting with three rule-based stakeholder agents, each holding a hidden target forecast they want adopted:

- **Demand Planner** — market-facing, pushes higher
- **Supply Planner** — capacity-facing, pushes lower
- **Finance** — budget-anchored, conservative

Over 4 turns — 2 stakeholder rounds and 2 LLM turns — the Consensus Planner must extract preferences from templated stakeholder messages, probe with a clarifying question or preliminary number, and commit to a final forecast that balances all three positions while staying grounded in procedurally generated true demand.

Turn structure is asymmetric:

| Turn | Who | What |
|---|---|---|
| 1 | Stakeholders | Demand → Supply → Finance each emit one opener. Later speakers see earlier ones. |
| 2 | **LLM (probe)** | Asks a clarifying question or floats a preliminary number. |
| 3 | Stakeholders | Each reacts to the LLM probe, revealing sharper signals. |
| 4 | **LLM (commit)** | Must emit `FINAL FORECAST: <number>` with justification referencing each stakeholder. |

---

## Reward structure

Five rubrics composed with a format gate:

| Rubric | Weight | What it measures |
|---|---|---|
| **Format (gate)** | — | Turn 4 must contain `FINAL FORECAST: <number>`. Failing this zeros the episode. |
| **Accuracy** | 0.40 | Commit vs. procedurally-generated true demand. Full credit at <=1% error, zero at >=8%. |
| **Consensus** | 0.30 | Commit vs. average of the 3 stakeholder targets. Full credit at <=1%, zero at >=6%. |
| **Extraction** | 0.20 | Did the final justification reference each stakeholder target (within 5%)? |
| **Efficiency** | 0.10 | Did turn 2 contain a genuine probe (question + number beats either beats neither)? |

Final reward = `format_gate * sum(weight * rubric_score)`, clamped to [0.01, 0.99].

The rubrics produce clean gradient signal — hand-tested on three representative episodes:

| Behavior | Reward |
|---|---|
| Ideal (probes actively, commits near true demand, cites all stakeholders) | ~0.97 |
| Lazy (minimal probe, commits at baseline, ignores stakeholders) | ~0.34 |
| Format fail (no FINAL FORECAST marker) | 0.01 |

---

## Tasks

Three difficulty levels, differing in stakeholder preference spread and noise:

| Task | Baseline | Prefs (Demand/Supply/Finance) | Noise |
|---|---|---|---|
| `task_easy` | 1000 | +8% / -2% / 0% | +/-30 |
| `task_medium` | 1500 | +18% / -4% / 0% | +/-80 |
| `task_hard` | 800 | +25% / -8% / 0% | +/-120 |

Hard scenarios require genuine trade-off reasoning — naive averaging is insufficient.

---

## Why this domain

S&OP is the monthly cross-functional business process that every consumer-goods company runs. It is exactly where AI agents need to land: a real-world task with structured disagreement, hidden preferences, and a forced decision under uncertainty. Unilever's real S&OP implementation drove a 20% reduction in supply chain waste — this environment simulates the mechanism behind outcomes like that.

No supply-chain environment currently exists in the OpenEnv Hub. This one fills that gap and is grounded in the author's direct consulting experience with Anaplan-based S&OP at EY.

---

## Training plan (onsite, April 25–26)

- **Model:** Qwen2.5-3B-Instruct, 4-bit via Unsloth
- **Algorithm:** GRPO (group-relative policy optimization) via HF TRL
- **Loop:** `rollout_func` pattern — env drives the 4-turn episode, TRL handles policy updates
- **Baseline:** untrained Qwen2.5-3B on same 3 tasks, same seeds, reward captured
- **Success metric:** trained agent reward on `task_hard` exceeds baseline by a visible margin, with monotonic improvement on the training reward curve

Plots, notebook, and final commentary will be committed to the repo during the onsite.

---

## Running locally

Build and run:

```
docker build -t sop-war-room:latest .
docker run --rm -p 8000:8000 sop-war-room:latest
```

Validate (in another terminal):

```
openenv validate --url http://localhost:8000
```

Inference against a running env:

```
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
export HF_TOKEN=<your-token>
export ENV_BASE_URL=http://localhost:8000
python inference.py
```

---

## OpenEnv contract compliance

Runtime validation passes 6/6 required criteria against `openenv-core` standard v0.1.0:

- `openapi_version_available`
- `health_endpoint`
- `metadata_endpoint`
- `schema_endpoint`
- `mcp_endpoint` (MCP tool names follow the rules — no reserved names used)
- `mode_endpoint_consistency` (`/reset`, `/step`, `/state`)

---

## Architecture

```
sop-war-room/
|-- models.py              # Pydantic: Observation, Action, State
|-- stakeholders.py        # Rule-based DemandPlanner, SupplyPlanner, Finance
|-- inference.py           # OpenAI-client-compatible driver (HF router / vLLM / any)
|-- mock_llm.py            # Local mock server for offline testing
|-- server/
|   |-- app.py             # FastAPI: health, metadata, schema, mcp, reset, step, state
|   |-- environment.py     # Episode engine - turn schedule, scenarios
|   +-- grader.py          # 5-rubric scorer with format gate
|-- blog/
|   +-- post.md            # Hugging Face blog post (in progress)
|-- plots/                 # Training reward/loss plots (populated onsite)
|-- openenv.yaml           # Env manifest (3 tasks)
|-- Dockerfile
|-- pyproject.toml         # openenv-core + deps, `server` entrypoint
+-- uv.lock
```

---

## Future work (v2)

- Stochastic stakeholder personalities (bluffing, priority drift)
- Multi-period episodes (rolling 18-month horizon, real S&OP cadence)
- Dashboard for replaying and grading human vs. agent transcripts
- Hooking to real Anaplan S&OP models for industrial-scale training data

---

## Author

Shubham Yeole — Anaplan / S&OP consultant (EY), currently transitioning to ML/AI via Scaler DSML.

- GitHub: @shubham565123
- Hugging Face: @shubhamyeole565

## License

Apache-2.0
