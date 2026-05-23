from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv
import sys
import os
import json
import concurrent.futures

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp_servers.market_data_mcp.server import get_price_history, get_financial_ratios
from mcp_servers.news_sentiment_mcp.server import search_news, score_sentiment
from rag.vector_store.embedder import store_news, retrieve

# ── CREWAI TOOLS wrapping MCP functions ──────────────────

@tool("Fetch Price History")
def fetch_price_history_tool(ticker: str) -> str:
    """Fetch the last month of OHLCV price history for a stock ticker."""
    result = get_price_history(ticker, "1mo")
    return json.dumps(result)

@tool("Fetch Financial Ratios")
def fetch_financial_ratios_tool(ticker: str) -> str:
    """Fetch key financial ratios (P/E, EPS, debt/equity, market cap) for a stock ticker."""
    result = get_financial_ratios(ticker)
    return json.dumps(result)

@tool("Search Stock News")
def search_news_tool(ticker: str) -> str:
    """Search recent news articles for a stock ticker (last 7 days)."""
    result = search_news(ticker, days_back=7)
    return json.dumps(result)

@tool("Score News Sentiment")
def score_sentiment_tool(headlines_json: str) -> str:
    """Score the sentiment of news headlines. Input: JSON string of a headlines list."""
    try:
        headlines = json.loads(headlines_json) if isinstance(headlines_json, str) else headlines_json
    except Exception:
        return json.dumps({"score": 0, "label": "neutral", "summary": "Could not parse headlines."})
    result = score_sentiment(headlines)
    return json.dumps(result)

@tool("RAG Context Retrieval")
def rag_retrieve_tool(query: str) -> str:
    """Retrieve relevant financial context from the vector store for a given query."""
    docs = retrieve(query, n_results=3)
    return "\n".join(docs) if docs else "No relevant context found."

# ── AGENT 1: Research Analyst (CrewAI) ───────────────────
def run_research_analyst(ticker: str) -> dict:
    """Runs the CrewAI Research Analyst for a given ticker."""
    print(f"\n Research Analyst (CrewAI) working on {ticker}...")

    # Pre-fetch and store news in RAG before CrewAI agent runs
    news_result = search_news(ticker, days_back=7)
    headlines = news_result.get("headlines", [])
    store_news(ticker, headlines)
    sentiment = score_sentiment(headlines)

    research_agent = Agent(
        role="Financial Research Analyst",
        goal=(
            f"Analyze recent news and market sentiment for {ticker} "
            "to identify key developments, risks, and investment themes."
        ),
        backstory=(
            "You are a seasoned financial research analyst with 15 years of experience "
            "at a top-tier investment bank. You specialize in synthesizing news, "
            "qualitative risk factors, and sentiment signals into actionable insights."
        ),
        tools=[search_news_tool, score_sentiment_tool, rag_retrieve_tool],
        verbose=False,
        allow_delegation=False
    )

    research_task = Task(
        description=(
            f"Research the stock ticker {ticker}. "
            f"Use the 'Search Stock News' tool to fetch recent headlines. "
            f"Use the 'Score News Sentiment' tool on those headlines. "
            f"Use the 'RAG Context Retrieval' tool with query '{ticker} stock news risk'. "
            f"Write a concise research summary covering: "
            f"1) Recent developments, 2) Market sentiment, 3) Key risks. "
            f"At the very end of your output, add a final line in exactly this format: "
            f"SENTIMENT_LABEL: positive OR SENTIMENT_LABEL: neutral OR SENTIMENT_LABEL: negative"
        ),
        expected_output=(
            "A structured research summary with three clearly labelled sections: "
            "Recent Developments, Market Sentiment, and Key Risks. "
            "Followed by a final line: SENTIMENT_LABEL: positive/neutral/negative"
        ),
        agent=research_agent
    )

    crew = Crew(
        agents=[research_agent],
        tasks=[research_task],
        process=Process.sequential,
        verbose=False
    )

    crew_result = crew.kickoff(inputs={"ticker": ticker})
    raw_output = crew_result.raw if hasattr(crew_result, "raw") else str(crew_result)

    # Extract analyst's holistic sentiment label and strip it from the summary
    analyst_label = None
    clean_lines = []
    for line in raw_output.splitlines():
        if line.strip().upper().startswith("SENTIMENT_LABEL:"):
            label = line.split(":", 1)[1].strip().lower()
            if label in ("positive", "neutral", "negative"):
                analyst_label = label
        else:
            clean_lines.append(line)
    summary = "\n".join(clean_lines).strip()

    # Override raw headline sentiment label with analyst's holistic assessment
    if analyst_label:
        sentiment = {**sentiment, "label": analyst_label}

    return {
        "ticker": ticker,
        "headlines": headlines,
        "sentiment": sentiment,
        "rag_context": retrieve(f"{ticker} stock news risk"),
        "research_summary": summary
    }

# ── AGENT 2: Quant Analyst (CrewAI) ──────────────────────
def run_quant_analyst(ticker: str) -> dict:
    """Runs the CrewAI Quant Analyst for a given ticker."""
    print(f"\n Quant Analyst (CrewAI) working on {ticker}...")

    # Pre-fetch data for the return dict
    ratios = get_financial_ratios(ticker)
    price_history = get_price_history(ticker, period="1mo")

    quant_agent = Agent(
        role="Quantitative Analyst",
        goal=(
            f"Analyze price history and financial ratios for {ticker} "
            "to produce a data-driven buy/hold/sell investment signal."
        ),
        backstory=(
            "You are a quantitative analyst with deep expertise in financial modeling "
            "and technical analysis. You use hard data — price history, P/E ratios, "
            "debt levels — to generate objective, evidence-based signals."
        ),
        tools=[fetch_price_history_tool, fetch_financial_ratios_tool],
        verbose=False,
        allow_delegation=False
    )

    quant_task = Task(
        description=(
            f"Perform a quantitative analysis of {ticker}. "
            f"Use the 'Fetch Financial Ratios' tool for valuation metrics. "
            f"Use the 'Fetch Price History' tool for recent price action. "
            f"Provide: 1) Valuation assessment, 2) Price trend analysis, "
            f"3) Buy/Hold/Sell signal with clear reasoning."
        ),
        expected_output=(
            "A quantitative analysis with three clearly labelled sections: "
            "Valuation Assessment, Price Trend Analysis, and a Buy/Hold/Sell signal."
        ),
        agent=quant_agent
    )

    crew = Crew(
        agents=[quant_agent],
        tasks=[quant_task],
        process=Process.sequential,
        verbose=False
    )

    crew_result = crew.kickoff(inputs={"ticker": ticker})
    analysis = crew_result.raw if hasattr(crew_result, "raw") else str(crew_result)

    return {
        "ticker": ticker,
        "ratios": ratios,
        "price_history": price_history,
        "quant_analysis": analysis
    }

# ── RUN BOTH IN PARALLEL ─────────────────────────────────
def run_crew(ticker: str) -> dict:
    """Run Research and Quant analysts in parallel using ThreadPoolExecutor."""
    print(f"\n Starting CrewAI analysts for {ticker}...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        research_future = executor.submit(run_research_analyst, ticker)
        quant_future = executor.submit(run_quant_analyst, ticker)

        research_result = research_future.result()
        quant_result = quant_future.result()

    print(f" Both CrewAI analysts done for {ticker}.")
    return {
        "ticker": ticker,
        "research": research_result,
        "quant": quant_result
    }

if __name__ == "__main__":
    result = run_crew("AAPL")
    print("\n RESEARCH SUMMARY:")
    print(result["research"]["research_summary"])
    print("\n QUANT ANALYSIS:")
    print(result["quant"]["quant_analysis"])
