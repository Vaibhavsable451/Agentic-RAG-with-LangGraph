# 03 - Agentic RAG with LangGraph

A stateful graph where the system **decides** what to do instead of blindly stuffing context:

```
 START -> retrieve -> grade_documents --(relevant)--> generate -> END
                          |
                     (none relevant)
                          v
                    rewrite_query -> retrieve   (max 2 loops, then "I don't know")
```

```bash
pip install -r requirements.txt && cp .env.example .env
python -m src.graph "What is the password rotation policy?"
python -m src.graph "Who won the 2022 World Cup?"   # out of scope -> graceful refusal
```
Why LangGraph? Explicit state, conditional edges, loops and easy observability.
