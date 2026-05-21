from fastmcp import FastMCP
from newsapi import NewsApiClient
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

mcp = FastMCP("news-sentiment-mcp")
newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── TOOL 1 ──────────────────────────────────────────────
@mcp.tool()
def search_news(query: str, days_back: int = 7) -> dict:
    """Search recent news articles for a given query."""
    from datetime import datetime, timedelta
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        language="en",
        sort_by="relevancy",
        page_size=5
    )
    
    headlines = [
        {
            "title": a["title"],
            "source": a["source"]["name"],
            "published": a["publishedAt"],
            "description": a["description"]
        }
        for a in articles.get("articles", [])
    ]
    
    return {"query": query, "headlines": headlines}

# ── TOOL 2 ──────────────────────────────────────────────
@mcp.tool()
def score_sentiment(headlines: list) -> dict:
    """Score sentiment of a list of headlines using OpenAI."""
    if not headlines:
        return {"score": 0, "label": "neutral", "summary": "No headlines provided."}
    
    headlines_text = "\n".join(
        [f"- {h['title']}" for h in headlines if h.get("title")]
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial sentiment analyst. "
                    "Given headlines, return a JSON with: "
                    "score (-1.0 to 1.0), label (positive/neutral/negative), "
                    "and a one sentence summary. Return only JSON."
                )
            },
            {
                "role": "user",
                "content": f"Analyze these headlines:\n{headlines_text}"
            }
        ]
    )
    
    import json
    raw = response.choices[0].message.content
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)

# ── RESOURCE ─────────────────────────────────────────────
@mcp.resource("news://feed/{ticker}")
def news_feed(ticker: str) -> str:
    """Get latest news feed for a ticker."""
    result = search_news(ticker, days_back=3)
    headlines = result.get("headlines", [])
    if not headlines:
        return f"No recent news found for {ticker}."
    lines = [f"- [{h['source']}] {h['title']}" for h in headlines]
    return f"Latest news for {ticker}:\n" + "\n".join(lines)

# ── PROMPT ───────────────────────────────────────────────
@mcp.prompt()
def summarize_risk_factors(ticker: str) -> str:
    """Structured prompt for extracting risk factors from news."""
    return (
        f"You are a risk analyst. Based on recent news about {ticker}, "
        f"identify and list: 1) key risk factors, 2) potential catalysts, "
        f"3) overall risk level (low/medium/high). Be concise and specific."
    )

if __name__ == "__main__":
    mcp.run()