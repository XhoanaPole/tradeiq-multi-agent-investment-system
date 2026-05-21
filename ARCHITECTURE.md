# Architecture Diagram — TradeIQ Multi-Agent Investment System

## System Flow

```mermaid
flowchart TD
    User([User enters stock ticker]) --> LG

    subgraph LG["LangGraph Orchestrator (graph.py)"]
        N1[Node 1: Run Analysts] --> N2[Node 2: Risk Manager]
        N2 --> N3[Node 3: Report Writer]
        N3 --> N4[Node 4: Evaluator]
        N4 -->|score < 7| N3
        N4 -->|score >= 7| N5[Node 5: Human Review]
        N5 -->|approved| N6[Node 6: Execute]
        N5 -->|rejected| N7[Node 7: Rejected]
        N5 -->|revise| N3
    end

    subgraph CREW["CrewAI — Parallel Analysts (analysts.py)"]
        RA[Research Analyst\nAgent + Task + Crew]
        QA[Quant Analyst\nAgent + Task + Crew]
    end

    subgraph OAI["OpenAI Agents SDK (risk_and_report.py)"]
        RM[Risk Manager\nAgent + Runner]
        RW[Report Writer\nAgent + Runner]
        EV[Evaluator\nAgent + Runner]
    end

    subgraph A2A["A2A Server — port 8001 (a2a_risk_server.py)"]
        AC[/.well-known/agent.json\nAgent Card]
        AE[/a2a\nRisk Endpoint]
        AH[/health]
    end

    subgraph MCP1["FastMCP — market-data-mcp"]
        T1["Tool: get_price_history()"]
        T2["Tool: get_financial_ratios()"]
        R1["Resource: market://snapshot/{ticker}"]
        P1["Prompt: analyze_technicals()"]
    end

    subgraph MCP2["FastMCP — news-sentiment-mcp"]
        T3["Tool: search_news()"]
        T4["Tool: score_sentiment()"]
        R2["Resource: news://feed/{ticker}"]
        P2["Prompt: summarize_risk_factors()"]
    end

    subgraph RAG["ChromaDB RAG Pipeline (embedder.py)"]
        EMB[text-embedding-3-small]
        VDB[(ChromaDB\nVector Store)]
    end

    subgraph HITL["Human-in-the-Loop"]
        UI[Web UI\nlocalhost:8000]
        CP[Checkpoint Logger\noutputs/checkpoints/]
    end

    N1 --> CREW
    RA --> MCP2
    RA --> RAG
    QA --> MCP1
    RAG --> EMB --> VDB
    N2 --> A2A
    A2A --> OAI
    N3 --> RW
    N4 --> EV
    N5 --> UI
    UI --> CP
    N6 --> CP
```

## Frameworks & Roles

| Framework | Role in System |
|---|---|
| **LangGraph** | Orchestrates the full pipeline via a stateful graph with conditional routing and retry loops |
| **CrewAI** | Runs Research and Quant analyst agents in parallel using Agent/Task/Crew abstractions |
| **OpenAI Agents SDK** | Powers Risk Manager, Report Writer, and Evaluator via Agent + Runner pattern |
| **FastMCP** | Exposes market data and news sentiment as MCP servers with Tools, Resources, and Prompts |
| **ChromaDB** | Stores and retrieves news embeddings for RAG-augmented research analysis |
| **A2A Framework** | Risk Manager runs as a standalone HTTP agent server with an agent card |
| **FastAPI** | Serves the web UI and exposes REST endpoints for the full pipeline |

## MCP Integration Points

| MCP Server | Primitive | Name | Used By |
|---|---|---|---|
| market-data-mcp | Tool | `get_price_history` | Quant Analyst (CrewAI) |
| market-data-mcp | Tool | `get_financial_ratios` | Quant Analyst (CrewAI) |
| market-data-mcp | Resource | `market://snapshot/{ticker}` | Available for external clients |
| market-data-mcp | Prompt | `analyze_technicals` | Available for external clients |
| news-sentiment-mcp | Tool | `search_news` | Research Analyst (CrewAI) |
| news-sentiment-mcp | Tool | `score_sentiment` | Research Analyst (CrewAI) |
| news-sentiment-mcp | Resource | `news://feed/{ticker}` | Available for external clients |
| news-sentiment-mcp | Prompt | `summarize_risk_factors` | Available for external clients |
