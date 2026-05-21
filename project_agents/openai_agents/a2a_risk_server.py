from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import uvicorn

load_dotenv()

app = FastAPI(title="Risk Manager A2A Server")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── A2A AGENT CARD ───────────────────────────────────────
# This is like a business card for the agent
# It tells other agents what this agent can do
@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name": "Risk Manager Agent",
        "description": "Evaluates investment risk for a given stock ticker.",
        "version": "1.0.0",
        "endpoint": "http://localhost:8001/a2a",
        "capabilities": ["risk_assessment"],
        "input_schema": {
            "ticker": "string",
            "quant_analysis": "string",
            "research_summary": "string",
            "sentiment": "object"
        }
    }

# ── REQUEST MODEL ────────────────────────────────────────
class RiskRequest(BaseModel):
    ticker: str
    quant_analysis: str
    research_summary: str
    sentiment: dict = {}

# ── A2A ENDPOINT ─────────────────────────────────────────
@app.post("/a2a")
def assess_risk(request: RiskRequest):
    """
    This is the A2A endpoint.
    Other agents call this like a phone number to get risk assessment.
    """
    print(f"\n A2A Call received for {request.ticker}")

    system_prompt = (
        "You are a senior risk manager at an investment firm. "
        "Evaluate investment risk and respond ONLY in JSON with keys: "
        "risk_level (low/medium/high), "
        "red_flags (list of strings), "
        "recommendation (proceed/caution/avoid), "
        "reasoning (string)."
    )

    user_prompt = (
        f"Evaluate risk for {request.ticker}:\n\n"
        f"Quant Analysis: {request.quant_analysis}\n\n"
        f"Research Summary: {request.research_summary}\n\n"
        f"Sentiment: {request.sentiment}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        result = {"raw_response": raw}

    print(f" A2A Risk assessment complete for {request.ticker}")
    return {"ticker": request.ticker, "risk_assessment": result}

# ── HEALTH CHECK ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "online", "agent": "Risk Manager"}

if __name__ == "__main__":
    print("\n Starting Risk Manager A2A Server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)