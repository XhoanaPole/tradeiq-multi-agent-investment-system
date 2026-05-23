from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os
import json
import httpx
import logging
from datetime import datetime

class _NoHealthFilter(logging.Filter):
    def filter(self, record):
        return "/api/health" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_NoHealthFilter())

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from project_agents.orchestrator.graph import build_graph
from hitl.checkpoint_handler import CheckpointHandler

app = FastAPI(title="TradeIQ API")

# ─────────────────────────────────────────────────────────
# STATIC FILES
# ─────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str
    feedback: Optional[str] = None
    state: Optional[dict] = None


class DecisionRequest(BaseModel):
    ticker: str
    decision: str
    feedback: str = ""
    brief: str = ""
    state: dict = {}


# ─────────────────────────────────────────────────────────
# ANALYZE
# ─────────────────────────────────────────────────────────

@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):

    try:

        graph = build_graph()

        config = {
            "configurable": {
                "thread_id": f"tradeiq-{request.ticker}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
        }

        if request.state:
            initial_state = {
                "ticker": request.ticker,
                "research_result": request.state.get("research_result"),
                "quant_result": request.state.get("quant_result"),
                "risk_result": request.state.get("risk_result"),
                "report_result": request.state.get("report_result"),
                "evaluation": request.state.get("evaluation"),
                "human_decision": "revise",
                "human_feedback": request.feedback
            }
        else:
            initial_state = {
                "ticker": request.ticker,
                "research_result": None,
                "quant_result": None,
                "risk_result": None,
                "report_result": None,
                "evaluation": None,
                "human_decision": None,
                "human_feedback": None
            }

        final_state = graph.invoke(initial_state, config=config)

        risk_assessment = (
            final_state.get("risk_result", {})
            .get("risk_assessment", {})
        )

        evaluation = final_state.get("evaluation", {})

        research = final_state.get("research_result", {})

        report = final_state.get("report_result", {})

        sentiment = research.get("sentiment", {})

        return {
            "success": True,
            "ticker": request.ticker,

            "risk_level": risk_assessment.get(
                "risk_level",
                "N/A"
            ),

            "red_flags": risk_assessment.get(
                "red_flags",
                []
            ),

            "recommendation": risk_assessment.get(
                "recommendation",
                "N/A"
            ),

            "reasoning": risk_assessment.get(
                "reasoning",
                ""
            ),

            "score": evaluation.get(
                "score",
                "N/A"
            ),

            "sentiment": (
                sentiment.get("label", "N/A")
                if isinstance(sentiment, dict)
                else (sentiment[0].get("label", "N/A") if isinstance(sentiment, list) and sentiment else "N/A")
            ),

            "brief": report.get(
                "investment_brief",
                ""
            ),

            "state": {
                "risk_result": final_state.get(
                    "risk_result",
                    {}
                ),

                "evaluation": final_state.get(
                    "evaluation",
                    {}
                ),

                "research_result": {
                    "sentiment": research.get(
                        "sentiment",
                        {}
                    ),

                    "research_summary": research.get(
                        "research_summary",
                        ""
                    )
                },

                "quant_result": final_state.get(
                    "quant_result",
                    {}
                ),

                "report_result": report
            }
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────
# DECISION
# ─────────────────────────────────────────────────────────

@app.post("/api/decision")
def decision(request: DecisionRequest):

    try:

        # Save approved brief
        if request.decision == "approved":

            os.makedirs("outputs", exist_ok=True)

            with open(
                f"outputs/{request.ticker}_brief.txt",
                "w",
                encoding="utf-8"
            ) as f:

                f.write(request.brief)

        # Save checkpoint
        CheckpointHandler().log_checkpoint(
            request.ticker,
            request.state,
            request.decision,
            request.feedback
        )

        return {
            "success": True,
            "decision": request.decision
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────

@app.get("/api/history")
def history():

    try:

        logs = []

        log_dir = "outputs/checkpoints"

        if os.path.exists(log_dir):

            for filename in os.listdir(log_dir):

                if filename.endswith(".json"):

                    filepath = os.path.join(
                        log_dir,
                        filename
                    )

                    with open(
                        filepath,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        try:
                            data = json.load(f)

                            if isinstance(data, list):
                                logs.extend(data)

                        except:
                            pass

        # Sort newest first
        logs.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )

        return {
            "success": True,
            "logs": logs
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────
# DELETE ALL HISTORY
# ─────────────────────────────────────────────────────────

@app.delete("/api/history")
def delete_all_history():

    try:

        log_dir = "outputs/checkpoints"

        deleted = 0

        if os.path.exists(log_dir):

            for filename in os.listdir(log_dir):

                if filename.endswith(".json"):

                    os.remove(
                        os.path.join(log_dir, filename)
                    )

                    deleted += 1

        return {
            "success": True,
            "deleted": deleted
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────
# DELETE SINGLE ENTRY
# ─────────────────────────────────────────────────────────

@app.delete("/api/delete-entry")
def delete_entry(ticker: str, timestamp: str):

    try:

        log_dir = "outputs/checkpoints"

        if not os.path.exists(log_dir):

            raise HTTPException(
                status_code=404,
                detail="No history found"
            )

        found = False

        for filename in os.listdir(log_dir):

            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(
                log_dir,
                filename
            )

            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as f:

                entries = json.load(f)

            original_len = len(entries)

            # Remove matching entry
            entries = [

                e for e in entries

                if not (

                    e.get("ticker") == ticker
                    and
                    e.get("timestamp", "") == timestamp

                )
            ]

            # Something deleted
            if len(entries) < original_len:

                found = True

                # Remove file if empty
                if len(entries) == 0:

                    os.remove(filepath)

                else:

                    with open(
                        filepath,
                        "w",
                        encoding="utf-8"
                    ) as f:

                        json.dump(
                            entries,
                            f,
                            indent=2
                        )

                break

        if not found:

            raise HTTPException(
                status_code=404,
                detail="Entry not found"
            )

        return {
            "success": True
        }

    except HTTPException:
        raise

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────
# CHART ENDPOINT
# ─────────────────────────────────────────────────────────

@app.get("/api/chart/{ticker}")
def get_chart(ticker: str):

    try:

        url = (
            f"https://query1.finance.yahoo.com/"
            f"v8/finance/chart/{ticker}"
            f"?range=1mo&interval=1d"
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        }

        with httpx.Client(timeout=10) as client:

            r = client.get(
                url,
                headers=headers
            )

            r.raise_for_status()

            data = r.json()

        result = data["chart"]["result"][0]

        quote = result["indicators"]["quote"][0]

        return {

            "success": True,
            "ticker": ticker,

            "timestamps": result.get(
                "timestamp",
                []
            ),

            "open": quote.get(
                "open",
                []
            ),

            "high": quote.get(
                "high",
                []
            ),

            "low": quote.get(
                "low",
                []
            ),

            "close": quote.get(
                "close",
                []
            ),

            "volume": quote.get(
                "volume",
                []
            )
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────

@app.get("/api/health")
def health():

    return {
        "status": "online"
    }


# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )