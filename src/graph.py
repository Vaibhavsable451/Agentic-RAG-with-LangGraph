import sys
from pathlib import Path
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from src.config import get_llm, get_embeddings

MAX_REWRITES = 2

class State(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    rewrites: int

class Grade(BaseModel):
    relevant: bool = Field(description="True if the passage helps answer the question")

def build_retriever():
    docs = []
    for f in Path("data").glob("*.txt"):
        docs += TextLoader(str(f), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings()).as_retriever(search_kwargs={"k": 4})

retriever = None
llm = None

def retrieve(state: State):
    return {"documents": retriever.invoke(state["question"])}

def grade_documents(state: State):
    grader = llm.with_structured_output(Grade)
    prompt = ChatPromptTemplate.from_template(
        "Question: {q}\n\nPassage: {p}\n\nIs the passage relevant to answering the question?")
    kept = [d for d in state["documents"]
            if (prompt | grader).invoke({"q": state["question"], "p": d.page_content}).relevant]
    return {"documents": kept}

def rewrite_query(state: State):
    prompt = ChatPromptTemplate.from_template(
        "Rewrite this question to be a better search query for a company handbook. Return only the query.\n{q}")
    new_q = (prompt | llm | StrOutputParser()).invoke({"q": state["question"]})
    return {"question": new_q, "rewrites": state["rewrites"] + 1}

def generate(state: State):
    if not state["documents"]:
        return {"generation": "I don't know based on the provided documents."}
    ctx = "\n\n".join(d.page_content for d in state["documents"])
    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY this context. If insufficient, say you don't know.\n\n{ctx}\n\nQ: {q}")
    return {"generation": (prompt | llm | StrOutputParser()).invoke({"ctx": ctx, "q": state["question"]})}

def route_after_grading(state: State):
    if state["documents"]:
        return "generate"
    return "generate" if state["rewrites"] >= MAX_REWRITES else "rewrite_query"

def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading,
                            {"generate": "generate", "rewrite_query": "rewrite_query"})
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", END)
    return g.compile()

if __name__ == "__main__":
    llm = get_llm(); retriever = build_retriever()
    q = " ".join(sys.argv[1:]) or "What is the password policy?"
    for step in build_graph().stream({"question": q, "documents": [], "generation": "", "rewrites": 0}):
        for node, upd in step.items():
            print(f"-> {node}", {k: (len(v) if isinstance(v, list) else v) for k, v in upd.items()})
    final = build_graph().invoke({"question": q, "documents": [], "generation": "", "rewrites": 0})
    print("\nANSWER:", final["generation"])
