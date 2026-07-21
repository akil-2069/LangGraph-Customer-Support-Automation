# Assignment 2: AI-Powered Customer Support Automation System

**Built with LangGraph**

**Name:** Akilan. M
**Reg. No: 23BCE2088


---

## Overview

ABC Technologies receives a high volume of daily support requests across product information, technical issues, billing, account management, and refunds — all currently handled manually. This project automates that workflow using a LangGraph-based multi-agent system that:

1. Accepts customer queries
2. Classifies the issue type (Sales, Technical, Billing, Account, or Memory Recall)
3. Routes the query to the correct specialized support agent
4. Retrieves relevant context from company documents using RAG
5. Remembers previous customer interactions via SQLite
6. Escalates high-risk requests to a human supervisor for approval
7. Generates a final, polished response for the customer

---

## Project Structure

```
customer-support-assistant/
├── main.py                    # Full source code — all 10 tasks
├── memory.db                  # SQLite database (auto-created on first run)
├── README.md
└── docs/                      # Knowledge base documents used by the RAG pipeline
    ├── company_policy.txt
    ├── pricing_guide.txt
    ├── technical_manual.txt
    └── faq_document.txt
```

---

## Architecture (Task 1 — LangGraph Workflow)

```
Customer query
      ↓
Intent classification node
      ↓
Conditional router
      ↓
  ┌───────────┬──────────────┬─────────────┬──────────────┐
Sales       Technical     Billing       Account     Memory recall
agent        agent         agent         agent       (skips routing)
  └───────────┴──────────────┴─────────────┘
              ↓
       RAG retrieval (per agent)
              ↓
      High-risk request?
       ┌──────┴───────┐
      Yes             No
       ↓               ↓
Human supervisor → Supervisor agent
   approval        (validates response)
       └──────┬────────┘
              ↓
       Final response
       (saved to SQLite memory)
```

High-risk categories requiring human approval: refund requests, subscription cancellation, account closure, compensation requests, escalation to management.

---

## State Structure (Task 2)

```python
class SupportState(TypedDict):
    customer_id: str
    customer_name: Optional[str]
    query: str
    intent: Optional[Literal["sales", "technical", "billing", "account", "memory_recall"]]
    retrieved_context: Optional[str]
    department_response: Optional[str]
    requires_approval: bool
    approval_status: Optional[Literal["pending", "approved", "rejected"]]
    conversation_history: List[dict]
    final_response: Optional[str]
```

---

## Tech Stack

- Python 3.10
- LangGraph (`StateGraph`, conditional edges, `END`)
- LangChain (`ChatPromptTemplate`, `langchain-ollama`, `langchain-community`)
- Ollama — local LLM backend
  - `llama3.1` — chat / reasoning model (classification, agents, supervisor)
  - `nomic-embed-text` — embedding model (RAG vector store)
- FAISS — vector store for document retrieval
- SQLite — conversation memory persistence

---

## Setup Instructions

### 1. Install Ollama
Download from [ollama.com](https://ollama.com) and install.

### 2. Start the Ollama server (if not already running in background)
```
ollama serve
```
If you see `bind: Only one usage of each socket address...`, Ollama is already running — skip this step.

### 3. Pull the required models
```
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 4. Install Python dependencies
```
pip install langgraph langchain-ollama langchain-core langchain-community langchain-text-splitters faiss-cpu
```

### 5. Run the system
```
cd customer-support-assistant
python main.py
```

This will:
- Build the FAISS vector store from the 4 documents in `docs/`
- Initialize `memory.db` (SQLite)
- Run all 5 demonstration queries in sequence
- Pause once for human-in-the-loop approval (Query 4 — refund request)

When prompted:
```
Supervisor decision — approve this response? (y/n):
```
type `y` to approve or `n` to reject and see the fallback response.

---

## Demonstration Queries (Task 10)

| # | Query | Expected path |
|---|---|---|
| 1 | "What are the pricing plans available for your software?" | Sales |
| 2 | "I forgot my account password." | Account |
| 3 | "My application crashes whenever I upload a file." | Technical Support |
| 4 | "I need a refund for my annual subscription." | Billing — requires human approval |
| 5 | "What was my previous support issue?" | Memory recall — no department routing |

---

## Knowledge Base Documents (Task 6 — RAG)

| Document | Content |
|---|---|
| `company_policy.txt` | Refund, cancellation, account closure, compensation, escalation, and data privacy policies |
| `pricing_guide.txt` | Subscription plans, pricing, add-ons, free trial, payment methods |
| `technical_manual.txt` | Installation, login troubleshooting, crash troubleshooting, configuration, browser support |
| `faq_document.txt` | Common customer questions across password resets, billing, team management, data export |

Documents are chunked (500 chars, 50 overlap), embedded with `nomic-embed-text`, and stored in an in-memory FAISS index rebuilt on each run.

---

## SQLite Memory (Task 7)

Schema (`conversation_history` table in `memory.db`):

```sql
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    customer_name TEXT,
    query TEXT NOT NULL,
    intent TEXT,
    response TEXT,
    timestamp TEXT NOT NULL
);
```

Every completed query is saved here. Memory-recall queries (e.g. "What was my previous issue?") retrieve all rows matching the customer's ID and answer using that history.

---

## Human-in-the-Loop (Task 8)

High-risk requests are detected via keyword matching against the query (refund, cancel, cancellation, close account, compensation, escalate, supervisor, legal). When flagged, the workflow routes to `human_approval_node`, which prints the proposed response and pauses for a console-based approve/reject decision before continuing to the supervisor.

---

## Supervisor Agent (Task 9)

The final node in the graph. Reviews the department agent's draft response, polishes tone and clarity while preserving factual content, and is responsible for persisting the completed interaction to SQLite memory.

---

## Notes

- All processing runs locally via Ollama — no external API calls or costs.
- The vector store is rebuilt fresh on every run (not persisted to disk) since the document set is small; for production use this would be cached.
- Intent classification combines LLM-based categorization with keyword-based high-risk detection for reliability.
