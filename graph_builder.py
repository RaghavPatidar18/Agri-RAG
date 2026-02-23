import os
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from psycopg_pool import ConnectionPool
from index_docs import get_retriever
from database import DB_URI

load_dotenv()

FAISS_INDEX_PATH = "./faiss_index"
DOCS_FOLDER = "./documents"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

class State(TypedDict):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: str
    evidence: List[str]
    retries: int
    isuse: str
    use_reason: str

# --- 1. Nodes & Edges (Adapted from your code) ---
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(..., description="True if external documents are needed.")

def decide_retrieval(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Return JSON: should_retrieve (boolean). True if answering requires company docs."),
        ("human", "Question: {question}")
    ])
    decision = llm.with_structured_output(RetrieveDecision).invoke(prompt.format_messages(question=state["question"]))
    return {"need_retrieval": decision.should_retrieve}

def route_after_decide(state: State):
    return "retrieve" if state["need_retrieval"] else "generate_direct"

def generate_direct(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using general knowledge. If it needs company info, say 'I don't know'."),
        ("human", "{question}")
    ])
    return {"answer": llm.invoke(prompt.format_messages(question=state["question"])).content}

def retrieve(state: State):
    retriever = get_retriever()
    if not retriever:
        return {"docs": []}
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(..., description="True ONLY if document directly relates to the question topic.")

def is_relevant(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Return JSON: is_relevant. True if document discusses the question topic."),
        ("human", "Question:\n{question}\n\nDoc:\n{document}")
    ])
    relevant_docs = []
    for doc in state.get("docs", []):
        decision = llm.with_structured_output(RelevanceDecision).invoke(
            prompt.format_messages(question=state["question"], document=doc.page_content)
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}

def route_after_relevance(state: State):
    return "generate_from_context" if state.get("relevant_docs") else "no_answer_found"

def generate_from_context(state: State):
    context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on context. Don't mention getting context."),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])
    return {"answer": llm.invoke(prompt.format_messages(question=state["question"], context=context)).content, "context": context}

def no_answer_found(state: State):
    return {"answer": "No answer found in the knowledge base.", "context": ""}

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

def is_sup(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Verify if ANSWER is supported by CONTEXT. Return JSON: issup, evidence."),
        ("human", "Question:\n{question}\nAnswer:\n{answer}\nContext:\n{context}")
    ])
    decision = llm.with_structured_output(IsSUPDecision).invoke(
        prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", ""))
    )
    return {"issup": decision.issup, "evidence": decision.evidence}

def route_after_issup(state: State):
    if state.get("issup") == "fully_supported" or state.get("retries", 0) >= 3:
        return "accept_answer"
    return "revise_answer"

def accept_answer(state: State): return {}

def revise_answer(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "FORMAT: quote-only answer. Use ONLY CONTEXT."),
        ("human", "Question:\n{question}\nAnswer:\n{answer}\nCONTEXT:\n{context}")
    ])
    return {"answer": llm.invoke(prompt.format_messages(question=state["question"], answer=state.get("answer",""), context=state.get("context",""))).content, "retries": state.get("retries", 0) + 1}

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str

def is_use(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Decide if ANSWER addresses QUESTION. Return JSON: isuse, reason."),
        ("human", "Question:\n{question}\nAnswer:\n{answer}")
    ])
    decision = llm.with_structured_output(IsUSEDecision).invoke(prompt.format_messages(question=state["question"], answer=state.get("answer", "")))
    return {"isuse": decision.isuse, "use_reason": decision.reason}

def route_after_isuse(state: State):
    if state.get("isuse") == "useful": return "END"
    if state.get("rewrite_tries", 0) >= 3: return "no_answer_found"
    return "rewrite_question"

class RewriteDecision(BaseModel):
    retrieval_query: str

def rewrite_question(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite QUESTION for vector retrieval. Return JSON: retrieval_query."),
        ("human", "Question:\n{question}\nPrev Query:\n{retrieval_query}")
    ])
    decision = llm.with_structured_output(RewriteDecision).invoke(
        prompt.format_messages(question=state["question"], retrieval_query=state.get("retrieval_query", ""))
    )
    return {"retrieval_query": decision.retrieval_query, "rewrite_tries": state.get("rewrite_tries", 0) + 1, "docs": [], "relevant_docs": [], "context": ""}

# --- 2. Build and Compile Graph ---
def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)
    g.add_node("is_relevant", is_relevant)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found", no_answer_found)
    g.add_node("is_sup", is_sup)
    g.add_node("accept_answer", accept_answer)
    g.add_node("revise_answer", revise_answer)
    g.add_node("is_use", is_use)
    g.add_node("rewrite_question", rewrite_question)

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges("decide_retrieval", route_after_decide, {"generate_direct": "generate_direct", "retrieve": "retrieve"})
    g.add_edge("generate_direct", END)
    g.add_edge("retrieve", "is_relevant")
    g.add_conditional_edges("is_relevant", route_after_relevance, {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"})
    g.add_edge("no_answer_found", END)
    g.add_edge("generate_from_context", "is_sup")
    g.add_conditional_edges("is_sup", route_after_issup, {"accept_answer": "is_use", "revise_answer": "revise_answer"})
    g.add_edge("revise_answer", "is_sup")
    g.add_conditional_edges("is_use", route_after_isuse, {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"})
    g.add_edge("rewrite_question", "retrieve")

    return g.compile(checkpointer=checkpointer)

# Manage Postgres connection pool for the checkpointer globally
pool = ConnectionPool(conninfo=DB_URI, max_size=20)