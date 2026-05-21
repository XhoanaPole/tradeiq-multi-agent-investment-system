from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

# ── CHECKPOINT LOGGER ────────────────────────────────────
class CheckpointHandler:
    """Handles logging and tracking of all HITL checkpoints."""

    def __init__(self, log_dir: str = "outputs/checkpoints"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_checkpoint(self, ticker: str, state: dict, decision: str, feedback: str = None):
        """Log a human decision checkpoint to file."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "ticker": ticker,
            "decision": decision,
            "feedback": feedback or "None",
            "risk_level": state.get("risk_result", {})
                               .get("risk_assessment", {})
                               .get("risk_level", "N/A"),
            "brief_score": state.get("evaluation", {}).get("score", "N/A"),
            "recommendation": state.get("risk_result", {})
                                   .get("risk_assessment", {})
                                   .get("recommendation", "N/A"),
            "brief": state.get("report_result", {}).get("investment_brief", "")
        }

        # Save to JSON log file
        log_path = os.path.join(self.log_dir, f"{ticker}_{timestamp[:10]}.json")
        logs = []
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"\n Checkpoint logged: {log_path}")
        return log_entry

    def get_logs(self, ticker: str) -> list:
        """Retrieve all checkpoint logs for a ticker."""
        logs = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith(ticker):
                with open(os.path.join(self.log_dir, filename), "r") as f:
                    logs.extend(json.load(f))
        return logs

    def print_summary(self, ticker: str):
        """Print a summary of all decisions made for a ticker."""
        logs = self.get_logs(ticker)
        if not logs:
            print(f"No checkpoint logs found for {ticker}.")
            return

        print(f"\n CHECKPOINT SUMMARY FOR {ticker}")
        print("─" * 50)
        for log in logs:
            print(f"   {log['timestamp']}")
            print(f"     Decision   : {log['decision'].upper()}")
            print(f"     Risk Level : {log['risk_level'].upper()}")
            print(f"     Brief Score: {log['brief_score']}/10")
            print(f"     Feedback   : {log['feedback']}")
            print()

if __name__ == "__main__":
    # Quick test
    handler = CheckpointHandler()
    mock_state = {
        "risk_result": {"risk_assessment": {"risk_level": "medium", "recommendation": "proceed"}},
        "evaluation": {"score": 8}
    }
    handler.log_checkpoint("AAPL", mock_state, "approved", "Looks good")
    handler.print_summary("AAPL")