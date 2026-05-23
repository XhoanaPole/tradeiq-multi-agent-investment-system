# TradeIQ — Investment Research & Advisory System

A Multi-Agent AI system that researches stocks, assesses risk, and generates
investment briefs, with a Human-in-the-Loop checkpoint before any action is taken.

## Architecture

- **LangGraph** — Orchestrates the entire agent pipeline and manages state
- **CrewAI** — Powers the Research and Quant Analyst agents (run in parallel)
- **OpenAI Agents SDK** — Powers the Risk Manager, Report Writer, and Evaluator agents
- **FastMCP** — Two custom MCP servers for market data and news sentiment
- **ChromaDB** — Vector store for a custom RAG pipeline
- **A2A Framework** — Risk Manager exposed as a standalone agent server

## Agents

| Agent | Framework | Role |
|---|---|---|
| Orchestrator | LangGraph | Manages state graph and routes tasks |
| Research Analyst | CrewAI | Fetches news, scores sentiment, RAG retrieval |
| Quant Analyst | CrewAI | Fetches price history and financial ratios |
| Risk Manager | OpenAI Agents SDK + A2A | Evaluates risk via standalone A2A server |
| Report Writer | OpenAI Agents SDK | Generates structured investment brief |
| Evaluator | OpenAI Agents SDK | Scores investment signal strength (fundamentals, sentiment, risk), loops back if score < 7 |

## MCP Servers

### market-data-mcp
- `get_price_history(ticker, period)` — OHLCV price data via yfinance
- `get_financial_ratios(ticker)` — P/E, EPS, debt/equity ratios
- `market://snapshot/{ticker}` — Live price snapshot resource
- `analyze_technicals` — Pre-built technical analysis prompt

### news-sentiment-mcp
- `search_news(query, days_back)` — Recent headlines via NewsAPI
- `score_sentiment(headlines)` — Sentiment scoring via GPT
- `news://feed/{ticker}` — Ticker news feed resource
- `summarize_risk_factors` — Pre-built risk extraction prompt

## Agentic Patterns

- **Parallelization** — Research and Quant analysts run simultaneously via ThreadPoolExecutor
- **Orchestrator-Workers** — LangGraph orchestrates all agent calls
- **Evaluator-Optimizer** — Investment signal scored 0-10 (fundamentals, sentiment, risk level); loops back to Report Writer if score < 7 (max 2 retries)
- **Prompt Chaining** — News → Sentiment → Risk → Brief → Recommendation
- **Human-in-the-Loop** — Human approves/rejects/revises before execution
- **RAG Pipeline** — News headlines embedded and stored in ChromaDB, retrieved at analysis time

## Project Structure

```
genaiproject/
├── project_agents/
│   ├── orchestrator/
│   │   └── graph.py            # LangGraph state graph
│   ├── crew/
│   │   └── analysts.py         # CrewAI Research + Quant agents
│   └── openai_agents/
│       ├── risk_and_report.py  # Risk Manager, Report Writer, Evaluator
│       ├── a2a_risk_server.py  # Standalone A2A server for Risk Manager
│       └── a2a_risk_client.py  # A2A client with local fallback
├── mcp_servers/
│   ├── market_data_mcp/
│   │   └── server.py           # FastMCP market data server
│   └── news_sentiment_mcp/
│       └── server.py           # FastMCP news sentiment server
├── rag/
│   └── vector_store/
│       └── embedder.py         # ChromaDB RAG pipeline
├── hitl/
│   └── checkpoint_handler.py   # Human decision logger
├── static/
│   └── index.html              # Web UI
├── outputs/                    # Generated briefs and checkpoint logs
├── api.py                      # FastAPI web server
├── main.py                     # CLI entry point
├── .env                        # API keys (never commit this)
└── README.md
```

## Setup Instructions

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Clone the repository
```bash
git clone <your-repo-url>
cd genaiproject
```

### 3. Create virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install langgraph crewai openai openai-agents fastmcp chromadb yfinance newsapi-python python-dotenv fastapi uvicorn httpx
```

### 5. Set up API keys
Create a `.env` file in the root:
```
OPENAI_API_KEY=your_openai_key_here
NEWSAPI_KEY=your_newsapi_key_here
```

## Running the System

### Option A — Web UI (recommended)

**Terminal 1** — Start the A2A Risk Manager server:
```bash
python project_agents/openai_agents/a2a_risk_server.py
```

**Terminal 2** — Start the FastAPI web server:
```bash
python api.py
```

Then open your browser at `http://localhost:8000`

### Option B — CLI

**Terminal 1** — Start the A2A Risk Manager server:
```bash
python project_agents/openai_agents/a2a_risk_server.py
```

**Terminal 2** — Run the CLI:
```bash
python main.py
```

Then enter a stock ticker when prompted (e.g. `AAPL`, `TSLA`, `MSFT`)

## How it works

1. You enter a stock ticker (web UI or CLI)
2. Research Analyst and Quant Analyst run in parallel
3. Risk Manager is called via the A2A server at `http://localhost:8001`
4. Report Writer generates a structured investment brief
5. Evaluator scores the brief out of 10 — loops back if score < 7
6. You review the brief and choose: Approve / Reject / Revise
7. Brief saved to `outputs/` folder and decision logged to `outputs/checkpoints/`

## Bonus — A2A Framework

The Risk Manager runs as a fully standalone agent server:

- Risk assessment endpoint: `http://localhost:8001/a2a`
- Agent card: `http://localhost:8001/.well-known/agent.json`
- Interactive API docs: `http://localhost:8001/docs`
- Health check: `http://localhost:8001/health`

If the A2A server is offline, the system automatically falls back to running the Risk Manager locally.
