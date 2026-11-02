# Project 60: Retrieval-Augmented Generation (RAG) Chatbot

RAG chatbot: chunking, BM25 index (save/load JSON), grounded answers with citations, abstains when nothing relevant is found. Optional `generator(prompt)` hook for any LLM.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the demo
```bash
python -m aiprojects.p60_rag_chatbot
```

## Run the tests
```bash
python -m unittest discover -s tests -t .
```

## Structure
- `aiprojects/p60_rag_chatbot.py` - the project (read the module docstring first)
- `aiprojects/common/` - small shared helpers (text utils, NumPy MLP, toy game environments)
- `tests/test_p60.py` - unit tests

Requires Python 3.10+ and NumPy only.
