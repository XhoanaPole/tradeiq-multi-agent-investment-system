from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional
from dotenv import load_dotenv
import sys
import os

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from project_agents.crew.analysts import run_crew
from project_agents.openai_agents.a2a_risk_client import call_risk_manager
from project_agents.openai_agents.risk_and_report import (
    run_report_writer,
    evaluate_brief
)

# ── STATE DEFINITION ─────────────────────────────────────
class InvestmentState(TypedDict):
    ticker: str
    research_result: Optional[dict]
    quant_result: Optional[dict]
    risk_result: Optional[dict]
    report_result: Optional[dict]
    evaluation: Optional[dict]
    human_decision: Optional[str]
    human_feedback: Optional[str]
    retry_count: int

# ── NODE 1: Run CrewAI Analysts ──────────────────────────
def run_analysts_node(state: InvestmentState) -> InvestmentState:
    if state.get("research_result") and state.get("quant_result"):
        print("\n═══════════════════════════════════════")
        print(f"  NODE 1: Skipping Analysts (using cached results)")
        print("═══════════════════════════════════════")
        return state

    print("\n═══════════════════════════════════════")
    print(f"  NODE 1: Running Analysts for {state['ticker']}")
    print("═══════════════════════════════════════")
    crew_result = run_crew(state["ticker"])
    return {
        **state,
        "research_result": crew_result["research"],
        "quant_result": crew_result["quant"]
    }

def run_risk_node(state: InvestmentState) -> InvestmentState:
    if state.get("risk_result"):
        print("\n═══════════════════════════════════════")
        print(f"  NODE 2: Skipping Risk Manager (using cached results)")
        print("═══════════════════════════════════════")
        return state

    print("\n═══════════════════════════════════════")
    print(f"  NODE 2: Running Risk Manager for {state['ticker']} via A2A")
    print("═══════════════════════════════════════")
    
    result = call_risk_manager(
        ticker=state["ticker"],
        quant_analysis=state["quant_result"].get("quant_analysis", ""),
        research_summary=state["research_result"].get("research_summary", ""),
        sentiment=state["research_result"].get("sentiment", {})
    )

    # Handle both A2A response format and local fallback format
    if "risk_assessment" not in result:
        result = {"ticker": state["ticker"], "risk_assessment": result}

    # Make sure risk_assessment has all required keys
    ra = result.get("risk_assessment", {})
    if "risk_level" not in ra:
        ra["risk_level"] = "medium"
    if "red_flags" not in ra:
        ra["red_flags"] = []
    if "recommendation" not in ra:
        ra["recommendation"] = "caution"
    if "reasoning" not in ra:
        ra["reasoning"] = "No reasoning provided."

    result["risk_assessment"] = ra
    return {**state, "risk_result": result}

# ── NODE 3: Run Report Writer ────────────────────────────
def run_report_node(state: InvestmentState) -> InvestmentState:
    print("\n═══════════════════════════════════════")
    print(f"  NODE 3: Writing Investment Brief for {state['ticker']}")
    print("═══════════════════════════════════════")
    report_result = run_report_writer(
        state["ticker"],
        state["research_result"],
        state["quant_result"],
        state["risk_result"],
        state.get("human_feedback", "")
    )
    return {**state, "report_result": report_result}

# ── NODE 4: Evaluator ────────────────────────────────────
def run_evaluator_node(state: InvestmentState) -> InvestmentState:
    print("\n═══════════════════════════════════════")
    print(f"  NODE 4: Evaluating Brief Quality")
    print("═══════════════════════════════════════")
    brief = state["report_result"]["investment_brief"]
    evaluation = evaluate_brief(brief)
    return {**state, "evaluation": evaluation}

def human_review_node(state: InvestmentState) -> InvestmentState:
    print("\n═══════════════════════════════════════")
    print("  NODE 5: ⏸  HUMAN REVIEW CHECKPOINT")
    print("═══════════════════════════════════════")
    print(f"\n Brief ready for {state['ticker']}")
    print(f"  Risk Level: {state['risk_result']['risk_assessment'].get('risk_level', 'N/A').upper()}")
    print(f" Brief Score: {state['evaluation'].get('score', 'N/A')}/10")
    print("\n Awaiting human decision via Streamlit UI...")
    return {**state, "human_decision": "approved"}

# ── NODE 6: Execute Recommendation ───────────────────────
def execute_node(state: InvestmentState) -> InvestmentState:
    print("\n═══════════════════════════════════════")
    print(f"  NODE 6:  Executing Recommendation")
    print("═══════════════════════════════════════")
    recommendation = state["risk_result"]["risk_assessment"].get("recommendation", "N/A")
    print(f"\n Recommendation for {state['ticker']}: {recommendation.upper()}")
    print(" Brief saved to output.")
    
    # Save brief to file
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{state['ticker']}_brief.txt", "w") as f:
        f.write(state["report_result"]["investment_brief"])
    
    return state

# ── NODE 7: Rejected ─────────────────────────────────────
def rejected_node(state: InvestmentState) -> InvestmentState:
    print("\n═══════════════════════════════════════")
    print(f"  NODE 7:  Recommendation Rejected")
    print("═══════════════════════════════════════")
    print(f"Reason: {state.get('human_feedback', 'No reason provided.')}")
    return state

# ── ROUTING LOGIC ─────────────────────────────────────────
def route_after_evaluation(state: InvestmentState) -> str:
    """If brief score is too low, go back to report writer."""
    score = state["evaluation"].get("score", 0)
    retry = state.get("retry_count", 0)
    if score < 7 and retry < 2:
        print(f"\n Score too low ({score}/10), retrying report... (attempt {retry+1})")
        return "retry"
    return "human_review"

def route_after_human(state: InvestmentState) -> str:
    """Route based on human decision."""
    decision = state.get("human_decision", "approved")
    if decision == "approved":
        return "execute"
    elif decision == "rejected":
        return "rejected"
    else:
        return "revise"

# ── BUILD THE GRAPH ───────────────────────────────────────
def build_graph():
    memory = MemorySaver()
    graph = StateGraph(InvestmentState)

    # Add nodes
    graph.add_node("run_analysts", run_analysts_node)
    graph.add_node("run_risk", run_risk_node)
    graph.add_node("run_report", run_report_node)
    graph.add_node("run_evaluator", run_evaluator_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("execute", execute_node)
    graph.add_node("rejected", rejected_node)

    # Add edges
    graph.set_entry_point("run_analysts")
    graph.add_edge("run_analysts", "run_risk")
    graph.add_edge("run_risk", "run_report")
    graph.add_edge("run_report", "run_evaluator")

    # Conditional: evaluator → retry or human review
    graph.add_conditional_edges(
        "run_evaluator",
        route_after_evaluation,
        {
            "retry": "run_report",
            "human_review": "human_review"
        }
    )

    # Conditional: human → execute, reject, or revise
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "execute": "execute",
            "rejected": "rejected",
            "revise": "run_report"
        }
    )

    graph.add_edge("execute", END)
    graph.add_edge("rejected", END)

    return graph.compile(checkpointer=memory)

if __name__ == "__main__":
    graph = build_graph()
    initial_state = {
        "ticker": "AAPL",
        "research_result": None,
        "quant_result": None,
        "risk_result": None,
        "report_result": None,
        "evaluation": None,
        "human_decision": None,
        "human_feedback": None,
        "retry_count": 0
    }
    config = {"configurable": {"thread_id": "session-1"}}
    graph.invoke(initial_state, config=config)