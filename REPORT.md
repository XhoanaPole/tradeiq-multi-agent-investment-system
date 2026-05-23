  # Project Report — TradeIQ Multi-Agent Investment System

## 1. Problem Statement

Investment research is a time-intensive process that requires synthesizing data from
multiple sources : price history, financial ratios, news sentiment, and risk factors ,
before a human decision-maker can act. Doing this manually for even a single stock
takes hours and is prone to cognitive bias.

**TradeIQ** automates this process using a coordinated pipeline of specialized AI agents,
each responsible for one part of the analysis. The system produces a structured
investment brief, evaluates its own quality, and presents it to a human for a final
approve/reject/revise decision , ensuring AI augments rather than replaces human judgment.

---

## 2. Design Rationale

### Why LangGraph as the orchestrator?
LangGraph provides a stateful graph model that makes the pipeline explicit and inspectable.
Each node is a discrete step, edges define the flow, and conditional routing allows
dynamic behavior , such as looping back to the report writer if the brief scores below 7.
The built-in `MemorySaver` checkpointer preserves state across the retry loop without
any custom implementation. No other framework offers this combination of explicitness
and flexibility for multi-step agent pipelines.

### Why CrewAI for the analysts?
CrewAI is purpose-built for role-based multi-agent collaboration. The `Agent`, `Task`,
and `Crew` abstractions map directly to the research domain: an agent has a role and
backstory, a task has a description and expected output, and a crew coordinates execution.
The two analyst agents (Research and Quant) run in parallel via `ThreadPoolExecutor`,
reducing total analysis time significantly compared to sequential execution.

### Why the OpenAI Agents SDK for Risk, Report, and Evaluator?
The OpenAI Agents SDK (`openai-agents`) provides a lightweight `Agent + Runner` pattern
that is well-suited for single-purpose agents that take structured input and return
structured output. The Risk Manager, Report Writer, and Evaluator each have a focused
instruction set and a predictable JSON or text output. The SDK handles the model call
lifecycle cleanly without the overhead of a full crew setup for single-agent tasks.

### Why FastMCP for data tools?
FastMCP implements the Model Context Protocol (MCP) standard, which defines a clean
separation between AI agents and the tools they use. By wrapping `yfinance` and
`NewsAPI` calls in MCP servers, the system follows the principle that agents should
not have direct dependencies on data sources. Each MCP server exposes Tools (callable
functions), Resources (queryable data), and Prompts (reusable prompt templates) ,
all three FastMCP primitives are implemented.

### Why ChromaDB for RAG?
ChromaDB is a lightweight, persistent vector store that runs locally without a separate
service. News headlines are embedded using OpenAI's `text-embedding-3-small` model and
stored before the Research Analyst runs. The analyst can then retrieve the most
semantically relevant context for its query, improving the quality of the research
summary beyond what a raw news search would provide.

### Why A2A for the Risk Manager?
The A2A (Agent-to-Agent) framework treats the Risk Manager as a standalone, independently
deployable agent. It exposes an agent card at `/.well-known/agent.json` (describing its
capabilities), a `/a2a` endpoint for receiving tasks, and a `/health` endpoint for
availability checks. This demonstrates real-world agent interoperability , the main
system calls the Risk Manager over HTTP as it would any external service, and falls back
to a local implementation if the server is offline.

---

## 3. MCP Implementation Details

Two custom MCP servers were built using FastMCP:

### market-data-mcp
Provides financial market data via the `yfinance` library.

- **Tool** `get_price_history(ticker, period)` — returns the last N days of OHLCV
  (Open, High, Low, Close, Volume) price data for any stock ticker.
- **Tool** `get_financial_ratios(ticker)` — returns key valuation metrics: P/E ratio,
  EPS, debt-to-equity ratio, market cap, 52-week high/low.
- **Resource** `market://snapshot/{ticker}` — a live price snapshot including current
  price, previous close, and open price.
- **Prompt** `analyze_technicals(ticker)` — a pre-built prompt template instructing
  an LLM to perform technical analysis on the price data.

### news-sentiment-mcp
Provides news retrieval and sentiment analysis via NewsAPI and OpenAI.

- **Tool** `search_news(query, days_back)` — fetches the 5 most relevant recent
  headlines for a given query from NewsAPI.
- **Tool** `score_sentiment(headlines)` — sends headlines to GPT-4o-mini and returns
  a structured sentiment score (-1.0 to 1.0), label (positive/neutral/negative),
  and a one-sentence summary.
- **Resource** `news://feed/{ticker}` — a formatted news feed showing the latest
  3 days of headlines for a ticker.
- **Prompt** `summarize_risk_factors(ticker)` — a pre-built prompt template for
  extracting risk factors from recent news.

Both MCP servers are imported directly as Python modules by the CrewAI agents,
which call their functions as `@tool`-decorated wrappers within the CrewAI framework.

---

## 4. Agentic Patterns Implemented

| Pattern | Implementation |
|---|---|
| **Orchestrator-Workers** | LangGraph graph orchestrates all 4 agents as sequential nodes |
| **Parallelization** | Research and Quant analysts run simultaneously via ThreadPoolExecutor |
| **Evaluator** | Evaluator scores the investment signal 0-10 (fundamentals, sentiment, risk level); score is shown to the human who decides to approve, reject, or revise |
| **Prompt Chaining** | News → Sentiment → RAG retrieval → Risk → Brief → Recommendation |
| **Human-in-the-Loop** | Human reviews the final brief via web UI and approves, rejects, or requests a revision |
| **RAG Pipeline** | News headlines embedded with text-embedding-3-small, stored and retrieved from ChromaDB |
| **A2A Communication** | Risk Manager exposed as a standalone HTTP agent with an agent card |

---

## 5. System Performance Evaluation

### Qualitative Observations

The system consistently produces structured, readable investment briefs covering
market sentiment, quantitative analysis, and risk assessment. The three-section
format (Executive Summary, Analysis, Recommendation) is maintained across all tickers.

### Evaluator Scores

During testing across multiple tickers (AAPL, TSLA, MSFT, NVDA, META, AMZN, JPM),
the Evaluator agent assigned scores based on investment signal strength , strong stocks
with positive sentiment and low risk scored 8-9/10, while speculative or high-risk tickers
scored 5-7/10. The retry loop triggered more frequently for high-risk tickers, producing
a revised brief with clearer risk disclosures on the second attempt.

### Latency

A full pipeline run takes approximately 45–90 seconds end-to-end, depending on:
- NewsAPI response time
- Number of LLM calls (4 agents + sentiment scoring + embedding)
- Whether the retry loop triggers

The parallelization of the two CrewAI analysts reduces this by running both
simultaneously rather than sequentially.

### Limitations

- **NewsAPI free tier** limits searches to articles from the past 30 days and
  returns a maximum of 5 articles per query.
- **RAG store** uses fixed IDs per ticker, so each run overwrites previous embeddings
  with the latest headlines ,this is intentional to keep the store fresh.
- **Human-in-the-Loop** is implemented at the API layer rather than via LangGraph's
  native `interrupt_before` mechanism. This is functionally equivalent for the web UI
  use case but means the graph does not natively pause mid-execution.

---

## 6. Conclusion

TradeIQ demonstrates how multiple AI frameworks can be composed into a single,
coherent multi-agent system. Each framework was chosen for a specific strength:
LangGraph for orchestration, CrewAI for role-based parallel analysis, OpenAI Agents SDK
for focused single-agent tasks, FastMCP for standardized tool exposure, ChromaDB for
retrieval-augmented generation, and A2A for agent interoperability. The result is a
system that automates the full investment research pipeline while keeping a human in
control of the final decision.
