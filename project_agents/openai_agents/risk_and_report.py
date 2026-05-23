from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
import os
import json

load_dotenv()

# ── AGENT 3: Risk Manager (OpenAI Agents SDK) ────────────
def run_risk_manager(ticker: str, quant_result: dict, research_result: dict) -> dict:
    """Evaluates portfolio-level risk using the OpenAI Agents SDK."""
    print(f"\n Risk Manager (OpenAI Agents SDK) evaluating {ticker}...")

    risk_agent = Agent(
        name="Risk Manager",
        instructions=(
            "You are a senior risk manager at an investment firm. "
            "Evaluate investment risk and respond ONLY in valid JSON with keys: "
            "risk_level (low/medium/high), "
            "red_flags (list of strings), "
            "recommendation (proceed/caution/avoid), "
            "reasoning (string)."
        ),
        model="gpt-4o-mini"
    )

    message = (
        f"Evaluate the risk for {ticker}:\n\n"
        f"Quant Analysis: {quant_result.get('quant_analysis', '')}\n\n"
        f"Research Summary: {research_result.get('research_summary', '')}\n\n"
        f"Sentiment: {research_result.get('sentiment', {})}\n\n"
        f"Financial Ratios: {quant_result.get('ratios', {})}"
    )

    result = Runner.run_sync(risk_agent, message)
    raw = result.final_output
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        risk_assessment = json.loads(clean)
    except json.JSONDecodeError:
        risk_assessment = {"raw_response": raw}

    print(f" Risk assessment complete: {risk_assessment.get('risk_level', 'unknown')} risk")
    return {"ticker": ticker, "risk_assessment": risk_assessment}


# ── AGENT 4: Report Writer (OpenAI Agents SDK) ───────────
def run_report_writer(
    ticker: str,
    research_result: dict,
    quant_result: dict,
    risk_result: dict,
    human_feedback: str = ""
) -> dict:
    """Synthesizes all findings into a structured investment brief."""
    print(f"\n Report Writer (OpenAI Agents SDK) generating brief for {ticker}...")

    report_agent = Agent(
        name="Report Writer",
        instructions=(
            "You are a professional investment report writer. "
            "Write clear, structured, and concise investment briefs "
            "that a portfolio manager can act on immediately. "
            "CRITICAL: Your Final Recommendation MUST align with the risk assessment provided. "
            "If risk recommendation is 'avoid' → Final Recommendation must be Sell / Avoid. "
            "If risk recommendation is 'caution' → Final Recommendation must be Hold. "
            "If risk recommendation is 'proceed' → Final Recommendation must be Buy. "
            "Never contradict the risk assessment in your conclusion. "
            "Never mention 'the Risk Manager', 'the Evaluator', or any internal agent by name. "
            "Write as if you are the sole author presenting your own analysis and conclusions."
        ),
        model="gpt-4o-mini"
    )

    risk_assessment = risk_result.get("risk_assessment", {})
    risk_recommendation = risk_assessment.get("recommendation", "caution")

    message = (
        f"Write a full investment brief for {ticker} using this data:\n\n"
        f"1. RESEARCH SUMMARY:\n{research_result.get('research_summary', '')}\n\n"
        f"2. QUANT ANALYSIS:\n{quant_result.get('quant_analysis', '')}\n\n"
        f"3. RISK ASSESSMENT:\n{json.dumps(risk_assessment, indent=2)}\n\n"
        f"IMPORTANT: The risk assessment conclusion is '{risk_recommendation.upper()}'. "
        f"Your Final Recommendation MUST be: "
        f"{'Buy' if risk_recommendation == 'proceed' else 'Hold' if risk_recommendation == 'caution' else 'Sell'}. "
        f"Do not override this. Do not mention where this conclusion came from.\n\n"
    )

    if human_feedback:
        message += (
            f"4. USER FEEDBACK FOR REVISION:\n"
            f"The user rejected the previous draft. You MUST address this feedback:\n"
            f"\"{human_feedback}\"\n\n"
        )

    message += (
        "Structure the brief with these sections:\n"
        "- Executive Summary\n"
        "- Market Sentiment\n"
        "- Quantitative Analysis\n"
        "- Risk Assessment\n"
        "- Final Recommendation (Buy/Hold/Sell aligned with the Risk Manager — include price target if possible)"
    )

    result = Runner.run_sync(report_agent, message)
    brief = result.final_output
    print(f" Investment brief generated for {ticker}.")
    return {"ticker": ticker, "investment_brief": brief}


# ── EVALUATOR (OpenAI Agents SDK) ────────────────────────
def evaluate_brief(brief: str, risk_result: dict = None, research_result: dict = None) -> dict:
    """Scores the investment signal strength of the brief."""
    print(f"\n Evaluator scoring the brief...")

    evaluator_agent = Agent(
        name="Signal Evaluator",
        instructions=(
            "You are a senior investment analyst scoring the strength of an investment opportunity from 0-10. "
            "Score based on: quality of fundamentals, sentiment direction, risk level, and recommendation clarity. "
            "Use this realistic scale: "
            "9-10 = exceptional opportunity — strong fundamentals, positive sentiment, low risk, clear Buy signal. "
            "7-8 = solid opportunity — good fundamentals, mostly positive sentiment, medium risk, reasonable upside. "
            "5-6 = mixed or uncertain — weak fundamentals, neutral/mixed sentiment, or elevated risk without clear upside. "
            "3-4 = poor opportunity — negative sentiment, high risk, multiple red flags. "
            "0-2 = avoid — very high risk, strongly negative outlook, or no clear investment case. "
            "Important: medium risk is normal for most stocks and should NOT lower the score on its own. "
            "A well-known large-cap stock with solid earnings and positive sentiment should score 7-9. "
            "Only score below 6 if there are genuine red flags, negative fundamentals, or negative sentiment. "
            "Respond ONLY in valid JSON with: "
            "score (0-10), passed (true/false if score >= 7), "
            "feedback (one sentence explaining the score)."
        ),
        model="gpt-4o-mini"
    )

    context = f"Investment Brief:\n\n{brief}"
    if risk_result:
        ra = risk_result.get("risk_assessment", {})
        context += (
            f"\n\nRisk Data:\n"
            f"- Risk level: {ra.get('risk_level', 'unknown')}\n"
            f"- Red flags: {ra.get('red_flags', [])}\n"
            f"- Risk recommendation: {ra.get('recommendation', 'unknown')}"
        )
    if research_result:
        sentiment = research_result.get("sentiment", {})
        if isinstance(sentiment, list):
            sentiment = sentiment[0] if sentiment else {}
        context += (
            f"\n\nSentiment Data:\n"
            f"- Score: {sentiment.get('score', 'unknown')} (scale: -1.0 negative to 1.0 positive)\n"
            f"- Label: {sentiment.get('label', 'unknown')}"
        )

    result = Runner.run_sync(evaluator_agent, context)
    raw = result.final_output
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        evaluation = json.loads(clean)
    except json.JSONDecodeError:
        evaluation = {"score": 0, "passed": False, "feedback": raw}

    print(f" Signal score: {evaluation.get('score', 'N/A')}/10 — Passed: {evaluation.get('passed', False)}")
    return evaluation


if __name__ == "__main__":
    mock_quant = {"quant_analysis": "Strong momentum, P/E ratio is fair.", "ratios": {"PE_ratio": 28}}
    mock_research = {"research_summary": "Positive news sentiment. New product launches.", "sentiment": {"score": 0.7}}
    mock_risk = run_risk_manager("AAPL", mock_quant, mock_research)
    mock_report = run_report_writer("AAPL", mock_research, mock_quant, mock_risk)
    print(mock_report["investment_brief"])
