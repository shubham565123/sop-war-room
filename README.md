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

![Status](https://img.shields.io/badge/status-active_development-green)
![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.1.0-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)
![Python](https://img.shields.io/badge/python-3.11+-blue)

A reinforcement-learning environment for training LLMs on the real enterprise task of **Sales & Operations Planning (S&OP)** — a monthly cross-functional meeting where sales, supply, and finance must agree on a single forecast despite hidden, competing preferences. Built on [Meta's OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

This environment isolates the *negotiation dynamic* at the heart of S&OP, not a full Anaplan planning cycle. The 4-turn structure focuses training on one well-scoped capability: extracting hidden preferences from stakeholders with competing agendas and committing to a balanced forecast under pressure.

---

## Links & Artifacts

| Artifact | Link |
|---|---|
| Hugging Face Space (live environment) | _to be added after deploy_ |
| GitHub repository | https://github.com/shubham565123/sop-war-room |
| Training notebook (Colab, TRL + Unsloth + Qwen2.5-3B GRPO) | _to be added after training run_ |
| Reward / loss plots | `plots/` (populated after training) |
| Blog post (Hugging Face Hub or in-repo) | _to be added (see `blog/post.md`)_ |

---

## Why S&OP

S&OP is the monthly cross-functional business process every consumer-goods company runs. A room full of functional owners with incompatible objectives — marketing wants aggressive growth, operations wants feasibility, finance wants budget adherence — must leave with one number everyone will be held accountable to.

This is an ideal domain for LLM agent research for three reasons:

1. **Structured disagreement with hidden information.** Each stakeholder holds a private preference and reveals it only through templated business language. The agent must infer, not be told.
2. **Forced decision under conflicting pressure.** There is no "ask for more time" escape. A number must be committed.
3. **Measurable ground truth.** Procedurally generated true demand provides unambiguous reward signal for forecast accuracy, while consensus and extraction scoring capture the social dimension.

To date, no supply-chain or planning environment exists in the OpenEnv Hub. This environment fills that gap and is grounded in the author's consulting experience with Anaplan-based S&OP implementations at EY.

---

## Environment design

### What the LLM sees

A Consensus Planner LLM sits at a simulated S&OP meeting with three rule-based stakeholder agents:

- **Demand Planner** — market-facing, hidden preference biased upward
- **Supply Planner** — capacity-facing, hidden preference biased downward
- **Finance** — budget-anchored, hidden preference near baseline

Each stakeholder holds a hidden target percentage. They reveal it only indirectly through templated messages that leak information gradually.

### Turn structure

Asymmetric — the LLM speaks twice, stakeholders speak six times. Eight messages per episode.

| Turn | Who | What |
|---|---|---|
| 1 | Stakeholders | Demand → Supply → Finance each emit one opener. Later speakers see earlier ones. |
| 2 | **LLM (probe)** | Asks a clarifying question or floats a preliminary number. |
| 3 | Stakeholders | Each reacts to the LLM probe, revealing sharper signals. |
| 4 | **LLM (commit)** | Must emit `FINAL FORECAST: <number>` with justification referencing each stakeholder. |

### Reward structure

Five rubrics composed with a format gate:

| Rubric | Weight | What it measures |
|---|---|---|
| **Format (gate)** | — | Turn 4 must contain `FINAL FORECAST: <number>`. Failing zeros the episode. |
| **Accuracy** | 0.40 | Commit vs. procedurally-generated true demand. Full credit at <=1% error, zero at >=8%. |
| **Consensus** | 0.30 | Commit vs. average of the 3 stakeholder targets. Full credit at <=1%, zero at >=6%. |
| **Extraction** | 0.20 | Did the final justification reference each stakeholder target (within 5%)? |
| **Efficiency** | 0.10 | Did turn 2 contain a genuine probe (question + number beats either beats neither)? |

Final reward = `format_gate * sum(weight * rubric_score)`, clamped to `[0.01, 0.99]`.

Hand-tested on three representative policies:

| Behavior | Reward |
|---|---|
| Ideal (probes actively, commits near true demand, cites all stakeholders) | ~0.97 |
| Lazy (minimal probe, commits at baseline, ignores stakeholders) | ~0.34 |
| Format fail (no FINAL FORECAST marker) | 0.01 |

Three orders of magnitude between failure and good behavior — clean gradient signal for GRPO.

### Tasks

Three difficulty levels, varying in stakeholder preference spread and ground-truth noise:

| Task | Baseline | Prefs (Demand/Supply/Finance) | Noise |
|---|---|---|---|
| `task_easy` | 1000 | +8% / -2% / 0% | +/-30 |
| `task_medium` | 1500 | +18% / -4% / 0% | +/-80 |
| `task_hard` | 800 | +25% / -8% / 0% | +/-120 |

Hard scenarios require genuine trade-off reasoning — naive averaging is insufficient.

---

## Training

- **Model:** Qwen2.5-3B-Instruct, 4-bit via Unsloth
- **Algorithm:** GRPO (group-relative policy optimization) via Hugging Face TRL
- **Loop:** `rollout_func` pattern — env drives the 4-turn episode, TRL handles policy updates
- **Baseline:** untrained Qwen2.5-3B on same 3 tasks, same seeds
- **Target:** trained agent reward on `task_hard` exceeds baseline by a visible margin, with monotonic improvement on the training reward curve

Notebook, plots, and commentary are committed to the repo as training proceeds.

---

## Running locally

Build and run:

```
docker build -t sop-war-room:latest .
docker run --rm -p 8000:8000 sop-war-room:latest
```

Validate against the OpenEnv v0.1.0 contract:

```
openenv validate --url http://localhost:8000
```

Expected: `passed: True | 6 / 6` on all required criteria.

Run an LLM against the environment:

```
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
export HF_TOKEN=<your-token>
export ENV_BASE_URL=http://localhost:8000
python inference.py
```

---

## OpenEnv contract compliance

Runtime validation passes all required criteria against `openenv-core` standard v0.1.0:

- `openapi_version_available`
- `health_endpoint`
- `metadata_endpoint`
- `schema_endpoint`
- `mcp_endpoint` (MCP tool names are non-reserved)
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
|   +-- post.md            # Write-up on design and results
|-- plots/                 # Training reward/loss plots
|-- openenv.yaml           # Env manifest (3 tasks)
|-- Dockerfile
|-- pyproject.toml         # openenv-core + deps, `server` entrypoint
+-- uv.lock
```

---

## Roadmap

- Stochastic stakeholder personalities (bluffing, priority drift, coalitional behavior)
- Multi-period episodes on a rolling 18-month horizon, matching real S&OP cadence
- SKU-level negotiation instead of single-aggregate forecasts
- Integration with real Anaplan model exports for industrial-scale training data
- Dashboard for side-by-side replay of human vs. agent transcripts

Contributions and issues are welcome.

---

## Author

**Shubham Yeole** — S&OP and supply-chain planning consultant with deep experience in Anaplan-based enterprise forecasting, now working at the intersection of planning systems and LLM agents. This environment draws directly on patterns observed across live S&OP implementations.

- GitHub: [@shubham565123](https://github.com/shubham565123)
- Hugging Face: [@shubhamyeole565](https://huggingface.co/shubhamyeole565)

---

## Origins

This environment was initially developed for the Meta × PyTorch × Hugging Face OpenEnv Hackathon (India, 2026), Theme 1 (Multi-Agent Interactions). It has since evolved into an ongoing exploration of reinforcement learning applied to enterprise planning workflows.

## License

Apache-2.0

## Citation

If you use this environment in your work, please cite it using the metadata in `CITATION.cff`, or:

```
Yeole, Shubham. (2026). S&OP War Room: A multi-agent reinforcement learning
environment for training LLMs on Sales & Operations Planning negotiation.
https://github.com/shubham565123/sop-war-room
```
