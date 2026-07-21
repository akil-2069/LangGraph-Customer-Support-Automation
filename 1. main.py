import os
import sqlite3
from datetime import datetime
from typing import TypedDict, Optional, List, Literal

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END


# ======================================================================
# TASK 2 — STATE STRUCTURE
# ======================================================================
# This TypedDict is the shared state object that flows through every
# node in the LangGraph. Each node reads from it and writes updates
# back into it.

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


# ======================================================================
# SHARED LLM INSTANCE
# ======================================================================
llm = ChatOllama(model="llama3.1", temperature=0)


# ======================================================================
# TASK 7 — SQLITE MEMORY
# ======================================================================
DB_PATH = "memory.db"


def init_db():
    """Creates the conversation_history table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            customer_name TEXT,
            query TEXT NOT NULL,
            intent TEXT,
            response TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_to_memory(customer_id: str, customer_name: str, query: str, intent: str, response: str):
    """Stores one conversation turn into SQLite memory."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO conversation_history
           (customer_id, customer_name, query, intent, response, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (customer_id, customer_name, query, intent, response, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_customer_history(customer_id: str) -> List[dict]:
    """Retrieves all past conversation turns for a given customer."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT query, intent, response, timestamp
           FROM conversation_history
           WHERE customer_id = ?
           ORDER BY timestamp ASC""",
        (customer_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"query": r[0], "intent": r[1], "response": r[2], "timestamp": r[3]}
        for r in rows
    ]


# ======================================================================
# TASK 6 — RAG PIPELINE
# ======================================================================
DOCS_DIR = "docs"
VECTORSTORE = None


def build_vectorstore():
    """
    Loads all .txt knowledge base documents, splits them into chunks,
    embeds them, and builds a FAISS vector store for retrieval.
    """
    global VECTORSTORE

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    all_chunks = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk, metadata={"source": filename}))

    VECTORSTORE = FAISS.from_documents(all_chunks, embeddings)
    print(f"[RAG] Vector store built with {len(all_chunks)} chunks from {DOCS_DIR}/")


def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieves the top-k most relevant document chunks for a query."""
    if VECTORSTORE is None:
        return ""
    results = VECTORSTORE.similarity_search(query, k=k)
    context_parts = []
    for doc in results:
        print(f"[RAG Retrieved] Source: {doc.metadata['source']} → {doc.page_content[:80]}...")
        context_parts.append(f"[Source: {doc.metadata['source']}]\n{doc.page_content}")
    return "\n\n".join(context_parts)
# ======================================================================
# TASK 3 — INTENT CLASSIFICATION NODE
# ======================================================================
HIGH_RISK_KEYWORDS = [
    "refund", "cancel", "cancellation", "close my account", "close account",
    "compensation", "escalate", "speak to manager", "speak with management",
    "supervisor", "legal"
]


def classify_intent(state: SupportState) -> SupportState:
    """
    Classifies the customer query into one of: sales, technical, billing,
    account, or memory_recall. Uses a simple LLM-based classification prompt.
    """
    query = state["query"]

    # Check first if this is a memory-recall question — no department needed
    memory_phrases = ["previous issue", "what was my", "last time", "earlier issue", "before"]
    if any(phrase in query.lower() for phrase in memory_phrases):
        state["intent"] = "memory_recall"
        print(f"[Intent Classifier] Query classified as: memory_recall")
        return state

    classification_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an intent classifier for a customer support system. "
         "Classify the customer query into EXACTLY ONE of these categories: "
         "sales, technical, billing, account. "
         "Sales = product info, plans, pricing. "
         "Technical = errors, installation, login, configuration issues. "
         "Billing = invoices, payments, refunds. "
         "Account = password reset, profile updates, activation/deactivation. "
         "Respond with ONLY the single category word, nothing else."),
        ("human", "{query}"),
    ])

    chain = classification_prompt | llm
    result = chain.invoke({"query": query})
    intent_raw = result.content.strip().lower()

    # Normalize the model's output to one of our valid categories
    valid_intents = ["sales", "technical", "billing", "account"]
    intent = next((i for i in valid_intents if i in intent_raw), "technical")

    state["intent"] = intent

    # Determine if this request needs human approval based on keywords
    if any(keyword in query.lower() for keyword in HIGH_RISK_KEYWORDS):
        state["requires_approval"] = True
        state["approval_status"] = "pending"
    else:
        state["requires_approval"] = False
        state["approval_status"] = None

    print(f"[Intent Classifier] Query classified as: {intent} "
          f"(requires_approval={state['requires_approval']})")
    return state


# ======================================================================
# TASK 4 — CONDITIONAL ROUTING
# ======================================================================

def route_by_intent(state: SupportState) -> str:
    """
    Conditional edge function. Returns the name of the next node to
    execute based on the classified intent.
    """
    intent = state["intent"]
    if intent == "memory_recall":
        return "memory_recall_node"
    elif intent == "sales":
        return "sales_agent"
    elif intent == "technical":
        return "technical_agent"
    elif intent == "billing":
        return "billing_agent"
    elif intent == "account":
        return "account_agent"
    return "technical_agent"  # fallback


# ======================================================================
# TASK 5 — SPECIALIZED DEPARTMENT AGENTS
# ======================================================================

def _run_department_agent(state: SupportState, department: str) -> SupportState:
    """Shared logic used by all four department agents."""
    query = state["query"]
    context = retrieve_context(query)
    state["retrieved_context"] = context

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are the {department} Support agent for ABC Technologies. "
         f"Answer the customer's question accurately and concisely using the "
         f"retrieved company document context below. If the context doesn't "
         f"fully cover the question, answer using general best judgment but "
         f"stay consistent with the context provided.\n\n"
         f"Retrieved context:\n{context}"),
        ("human", "{query}"),
    ])

    chain = agent_prompt | llm
    result = chain.invoke({"query": query})
    state["department_response"] = result.content
    print(f"[{department} Agent] Response generated.")
    return state


def sales_agent(state: SupportState) -> SupportState:
    return _run_department_agent(state, "Sales")


def technical_agent(state: SupportState) -> SupportState:
    return _run_department_agent(state, "Technical")


def billing_agent(state: SupportState) -> SupportState:
    return _run_department_agent(state, "Billing")


def account_agent(state: SupportState) -> SupportState:
    return _run_department_agent(state, "Account")


def memory_recall_node(state: SupportState) -> SupportState:
    """
    Handles memory-recall queries by pulling the customer's past
    conversation history from SQLite and answering using that history,
    without routing to any department.
    """
    history = get_customer_history(state["customer_id"])
    state["conversation_history"] = history

    if not history:
        state["department_response"] = (
            "I don't see any previous support history on file for you yet."
        )
        return state

    history_text = "\n".join(
        f"- [{h['intent']}] Q: {h['query']} | A: {h['response']}" for h in history
    )

    recall_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support assistant. Use the customer's past "
         "conversation history below to answer their question about "
         "previous interactions. Be specific and reference the actual "
         "past issue.\n\nConversation history:\n" + history_text),
        ("human", "{query}"),
    ])

    chain = recall_prompt | llm
    result = chain.invoke({"query": state["query"]})
    state["department_response"] = result.content
    print("[Memory Recall] Answered using stored conversation history.")
    return state


# ======================================================================
# TASK 8 — HUMAN-IN-THE-LOOP APPROVAL
# ======================================================================

def human_approval_node(state: SupportState) -> SupportState:
    """
    Simulates human-in-the-loop approval for high-risk requests
    (refunds, cancellations, account closure, compensation, escalation).

    In a real deployment this would pause execution and wait for a
    human supervisor's decision via a dashboard/API callback. Here we
    simulate that decision point with a console prompt so the approval
    step is genuinely interactive and demonstrable.
    """
    print("\n" + "!" * 70)
    print("HUMAN-IN-THE-LOOP APPROVAL REQUIRED")
    print(f"Customer query: {state['query']}")
    print(f"Proposed response: {state['department_response']}")
    print("!" * 70)

    decision = input("Supervisor decision — approve this response? (y/n): ").strip().lower()

    if decision == "y":
        state["approval_status"] = "approved"
        print("[Human Approval] Approved by supervisor.")
    else:
        state["approval_status"] = "rejected"
        state["department_response"] = (
            "Your request requires further review by our support team. "
            "A specialist will follow up with you within 1-2 business days."
        )
        print("[Human Approval] Rejected by supervisor — response replaced.")

    return state


def needs_approval(state: SupportState) -> str:
    """Conditional edge: routes to human approval if flagged, else skips to supervisor."""
    if state.get("requires_approval"):
        return "human_approval_node"
    return "supervisor_agent"


# ======================================================================
# TASK 9 — SUPERVISOR AGENT
# ======================================================================

def supervisor_agent(state: SupportState) -> SupportState:
    """
    Validates and polishes the department response before it's sent to
    the customer. Ensures tone is professional, accurate, and complete.
    """
    raw_response = state["department_response"]

    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the Supervisor agent for ABC Technologies customer support. "
         "Review the draft response below. Improve clarity, tone, and "
         "professionalism if needed, but keep it concise and keep all "
         "factual content unchanged. Output only the final response text "
         "to send to the customer."),
        ("human", "Draft response:\n{response}"),
    ])

    chain = supervisor_prompt | llm
    result = chain.invoke({"response": raw_response})
    state["final_response"] = result.content

    # Persist this turn to SQLite memory
    save_to_memory(
        customer_id=state["customer_id"],
        customer_name=state.get("customer_name", "Unknown"),
        query=state["query"],
        intent=state.get("intent", "unknown"),
        response=state["final_response"],
    )

    print("[Supervisor Agent] Final response validated and saved to memory.")
    return state


# ======================================================================
# TASK 1 — LANGGRAPH WORKFLOW CONSTRUCTION
# ======================================================================

def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("sales_agent", sales_agent)
    graph.add_node("technical_agent", technical_agent)
    graph.add_node("billing_agent", billing_agent)
    graph.add_node("account_agent", account_agent)
    graph.add_node("memory_recall_node", memory_recall_node)
    graph.add_node("human_approval_node", human_approval_node)
    graph.add_node("supervisor_agent", supervisor_agent)

    graph.set_entry_point("classify_intent")

    # Task 4 — conditional routing after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent",
            "memory_recall_node": "memory_recall_node",
        },
    )

    # After each department agent, check if human approval is needed
    for agent_node in ["sales_agent", "technical_agent", "billing_agent", "account_agent"]:
        graph.add_conditional_edges(
            agent_node,
            needs_approval,
            {
                "human_approval_node": "human_approval_node",
                "supervisor_agent": "supervisor_agent",
            },
        )

    # Memory recall bypasses approval and goes straight to supervisor
    graph.add_edge("memory_recall_node", "supervisor_agent")

    # After human approval, always go to supervisor
    graph.add_edge("human_approval_node", "supervisor_agent")

    # Supervisor is the final step
    graph.add_edge("supervisor_agent", END)

    return graph.compile()


# ======================================================================
# TASK 10 — DEMONSTRATION
# ======================================================================

DEMO_QUERIES = [
    {"customer_id": "C001", "customer_name": "Akilan",
     "query": "What are the pricing plans available for your software?"},
    {"customer_id": "C002", "customer_name": "David",
     "query": "I forgot my account password."},
    {"customer_id": "C003", "customer_name": "Priya",
     "query": "My application crashes whenever I upload a file."},
    {"customer_id": "C001", "customer_name": "Akilan",
     "query": "I need a refund for my annual subscription."},
    {"customer_id": "C001", "customer_name": "Akilan",
     "query": "What was my previous support issue?"},
]


def run_demo():
    print("\n" + "=" * 70)
    print("Initializing system...")
    print("=" * 70)

    init_db()
    build_vectorstore()

    app = build_graph()

    for i, item in enumerate(DEMO_QUERIES, 1):
        print("\n\n" + "#" * 70)
        print(f"QUERY {i}: {item['query']}")
        print(f"Customer: {item['customer_name']} ({item['customer_id']})")
        print("#" * 70)

        initial_state: SupportState = {
            "customer_id": item["customer_id"],
            "customer_name": item["customer_name"],
            "query": item["query"],
            "intent": None,
            "retrieved_context": None,
            "department_response": None,
            "requires_approval": False,
            "approval_status": None,
            "conversation_history": [],
            "final_response": None,
        }

        result = app.invoke(initial_state)

        print("\n" + "-" * 70)
        print(f"FINAL RESPONSE TO CUSTOMER:\n{result['final_response']}")
        print("-" * 70)


if __name__ == "__main__":
    run_demo()
