import requests
import json

A2A_SERVER_URL = "http://localhost:8001"

# ── CHECK IF SERVER IS ALIVE ─────────────────────────────
def check_server_health() -> bool:
    """Check if the A2A Risk Manager server is running."""
    try:
        response = requests.get(f"{A2A_SERVER_URL}/health", timeout=3)
        if response.status_code == 200:
            print(" A2A Risk Manager server is online!")
            return True
    except requests.exceptions.ConnectionError:
        print(" A2A Risk Manager server is offline!")
        return False

# ── GET AGENT CARD ───────────────────────────────────────
def get_agent_card() -> dict:
    """
    Fetch the agent's business card.
    This tells us what the agent can do before we call it.
    """
    try:
        response = requests.get(
            f"{A2A_SERVER_URL}/.well-known/agent.json",
            timeout=3
        )
        card = response.json()
        print(f"\n Agent Card received:")
        print(f"   Name        : {card['name']}")
        print(f"   Description : {card['description']}")
        print(f"   Endpoint    : {card['endpoint']}")
        return card
    except Exception as e:
        print(f" Could not fetch agent card: {e}")
        return {}

# ── CALL THE A2A RISK MANAGER ────────────────────────────
def call_risk_manager(
    ticker: str,
    quant_analysis: str,
    research_summary: str,
    sentiment: dict = {}
) -> dict:
    """
    Call the A2A Risk Manager server.
    This is like calling someone on the phone and asking for their opinion.
    """
    print(f"\n Calling A2A Risk Manager for {ticker}...")

    # First check if server is alive
    if not check_server_health():
        print("  Falling back to local risk manager...")
        from project_agents.openai_agents.risk_and_report import run_risk_manager
        return run_risk_manager(
            ticker,
            {"quant_analysis": quant_analysis},
            {"research_summary": research_summary, "sentiment": sentiment}
        )

    # Get the agent card first
    get_agent_card()

    # Make the A2A call
    payload = {
        "ticker": ticker,
        "quant_analysis": quant_analysis,
        "research_summary": research_summary,
        "sentiment": sentiment
    }

    try:
        response = requests.post(
            f"{A2A_SERVER_URL}/a2a",
            json=payload,
            timeout=30
        )
        result = response.json()
        print(f" A2A Response received for {ticker}")
        return result

    except Exception as e:
        print(f" A2A call failed: {e}")
        print("  Falling back to local risk manager...")
        from project_agents.openai_agents.risk_and_report import run_risk_manager
        return run_risk_manager(
            ticker,
            {"quant_analysis": quant_analysis},
            {"research_summary": research_summary, "sentiment": sentiment}
        )

if __name__ == "__main__":
    # Quick test
    result = call_risk_manager(
        ticker="AAPL",
        quant_analysis="Strong momentum, P/E ratio is 36, high debt.",
        research_summary="Positive news sentiment. Record earnings.",
        sentiment={"score": 0.7, "label": "positive"}
    )
    print("\n Risk Result:")
    print(json.dumps(result, indent=2))