from fastmcp import FastMCP
import yfinance as yf
from datetime import datetime

mcp = FastMCP("market-data-mcp")

# ── TOOL 1 ──────────────────────────────────────────────
@mcp.tool()
def get_price_history(ticker: str, period: str = "1mo") -> dict:
    """Fetch OHLCV price history for a given ticker."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        return {"error": f"No data found for {ticker}"}
    data = hist[["Open", "Close", "High", "Low", "Volume"]].tail(10)
    data.index = data.index.astype(str)
    return {
        "ticker": ticker,
        "period": period,
        "data": data.to_dict()
    }

# ── TOOL 2 ──────────────────────────────────────────────
@mcp.tool()
def get_financial_ratios(ticker: str) -> dict:
    """Fetch key financial ratios for a given ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "PE_ratio": info.get("trailingPE", "N/A"),
        "EPS": info.get("trailingEps", "N/A"),
        "debt_to_equity": info.get("debtToEquity", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
    }

# ── RESOURCE ─────────────────────────────────────────────
@mcp.resource("market://snapshot/{ticker}")
def market_snapshot(ticker: str) -> str:
    """Live price snapshot for a ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return (
        f"Ticker: {ticker}\n"
        f"Current Price: {info.get('currentPrice', 'N/A')}\n"
        f"Previous Close: {info.get('previousClose', 'N/A')}\n"
        f"Open: {info.get('open', 'N/A')}\n"
        f"Timestamp: {datetime.now().isoformat()}"
    )

# ── PROMPT ───────────────────────────────────────────────
@mcp.prompt()
def analyze_technicals(ticker: str) -> str:
    """Pre-built prompt for technical analysis interpretation."""
    return (
        f"You are a technical analyst. Analyze the recent price history "
        f"of {ticker} and identify: 1) trend direction, 2) support/resistance "
        f"levels, 3) any notable patterns. Be concise and data-driven."
    )

if __name__ == "__main__":
    mcp.run()