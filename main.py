import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from project_agents.orchestrator.graph import build_graph
from hitl.checkpoint_handler import CheckpointHandler

def main():
    print("\n")
    print("╔══════════════════════════════════════════════╗")
    print("║      Investment Research & Advisory System   ║")
    print("║        Multi-Agent AI — Powered by LLM       ║")
    print("╚══════════════════════════════════════════════╝")

    # Get ticker from user
    ticker = input("\nEnter a stock ticker to analyze (e.g. AAPL, TSLA, MSFT): ").strip().upper()
    if not ticker:
        print("No ticker entered. Defaulting to AAPL.")
        ticker = "AAPL"

    print(f"\n Starting analysis for {ticker}...\n")

    # Build and run the graph
    graph = build_graph()
    checkpoint_handler = CheckpointHandler()

    initial_state = {
        "ticker": ticker,
        "research_result": None,
        "quant_result": None,
        "risk_result": None,
        "report_result": None,
        "evaluation": None,
        "human_decision": None,
        "human_feedback": None
    }

    config = {"configurable": {"thread_id": f"session-{ticker}"}}

    # Run the graph
    final_state = graph.invoke(initial_state, config=config)

    # Log the human checkpoint
    if final_state.get("human_decision"):
        checkpoint_handler.log_checkpoint(
            ticker=ticker,
            state=final_state,
            decision=final_state["human_decision"],
            feedback=final_state.get("human_feedback")
        )

    # Final summary
    print("\n╔══════════════════════════════════════════════╗")
    print("║               SYSTEM COMPLETE                ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\n  Ticker   : {ticker}")
    print(f"  Decision : {final_state.get('human_decision', 'N/A').upper()}")
    print(f"  Risk     : {final_state.get('risk_result', {}).get('risk_assessment', {}).get('risk_level', 'N/A').upper()}")
    print(f"  Score    : {final_state.get('evaluation', {}).get('score', 'N/A')}/10")

    if final_state.get("human_decision") == "approved":
        print(f"\n Brief saved to: outputs/{ticker}_brief.txt")

    print("\n")
    checkpoint_handler.print_summary(ticker)

if __name__ == "__main__":
    main()