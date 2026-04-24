"""
Mock OpenAI-compatible server for offline smoke testing.

Run in a third terminal:
    uvicorn mock_llm:app --port 9000

Then set:
    export API_BASE_URL=http://localhost:9000/v1
    export API_KEY=dummy
    export MODEL_NAME=mock-model
"""
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.2
    max_tokens: int = 200


def generate_response(user_content: str) -> str:
    """Produce a plausible probe or commit based on prompt content."""
    # Extract baseline
    m = re.search(r"Baseline forecast:\s*([\d.]+)", user_content)
    baseline = float(m.group(1)) if m else 1000.0

    # Extract any numbers mentioned in the transcript (stakeholder targets)
    transcript_part = user_content.split("--- Meeting transcript so far ---")[-1]
    nums = [float(n.replace(",", "")) for n in
            re.findall(r"\b(\d{2,5}(?:,\d{3})*(?:\.\d+)?)\b", transcript_part)
            if 100 < float(n.replace(",", "")) < 10000]

    if "FINAL COMMIT" in user_content:
        # Commit turn: average the stakeholder numbers if present, else baseline
        if nums:
            avg = sum(nums) / len(nums)
            nums_str = ", ".join(f"{n:.0f}" for n in nums[:3])
            return (f"Balancing the positions I heard ({nums_str}), "
                    f"I propose a number between the demand-side push and the "
                    f"supply-side brake, close to the finance anchor. "
                    f"FINAL FORECAST: {avg:.0f}")
        return f"Based on the discussion, FINAL FORECAST: {baseline:.0f}"

    # Probe turn: ask a question + float a number
    probe_num = baseline * 1.04
    return (f"Thanks for the context. What if we aligned around {probe_num:.0f} "
            f"as a starting point — would that work for each of you?")


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> Dict[str, Any]:
    last_user_msg = ""
    for m in req.messages:
        if m.role == "user":
            last_user_msg = m.content
    response_text = generate_response(last_user_msg)
    return {
        "id": "mock-completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
    }
