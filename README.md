# Agentic Customer Support RAG Chatbot

An intelligent customer support chatbot built with **LangChain**, **FAISS**, and **Streamlit**. It combines Retrieval-Augmented Generation (RAG) over a Markdown knowledge base with direct tool execution (e.g., mock order lookup) and confidence-based fallback routing to ensure zero hallucinations.

---

## Key Features

- **Grounded KB Retrieval**: Chunks and embeds store policies using local `sentence-transformers` (`all-MiniLM-L6-v2`) or OpenAI embeddings with FAISS vector similarity search.
- **Confidence Scoring & Honest Escalation**: Automatically calculates cosine similarity scores. If top retrieval confidence is below the agent threshold, the agent escalates to a human rather than guessing.
- **Tool Execution & Parameter Validation**: Detects order tracking intents and invokes order lookup tools. Strictly validates parameter requirements (e.g. requesting missing Order IDs rather than hallucinating parameters).
- **Out-of-Scope Detection**: Filters non-customer-support queries (e.g., general knowledge, code generation, off-topic prompts) directly to escalation.
- **Interactive Web Interface**: Streamlit UI featuring decision badges (`KB RETRIEVED`, `TOOL EXECUTED`, `CLARIFICATION REQUESTED`, `ESCALATED`), expandable reasoning logs, and preset query quick-buttons.
- **Automated Evaluation Suite**: Benchmark runner (`eval/run_eval.py`) to measure query accuracy, intent routing, latency, and hallucination rates across standard customer test suites.

---

## Architecture Overview

```
                        +----------------------+
                        |   User Query (UI)    |
                        +----------+-----------+
                                   |
                                   v
                        +----------------------+
                        |    Intent Router     |
                        +--+--------+-------+--+
                           |        |       |
            +--------------+        |       +----------------+
            | (Tool Needed)         | (KB Question)          | (Out of Scope / Ambiguous)
            v                       v                        v
  +------------------+    +--------------------+   +-------------------+
  |  Parameter Check |    | Vector Search      |   | Escalate / Request|
  | (Extract Ord ID) |    | (FAISS Index)      |   | Clarification     |
  +--------+---------+    +---------+----------+   +-------------------+
           |                        |
     +-----+-----+                  v
     |           |          Confidence >= threshold?
  (Found)   (Missing)      /                   \
     |           |       (Yes)                 (No)
     v           v         v                    v
 [Execute]  [Request ID] [LLM/Grounded]     [Escalate to]
 [ Tool  ]               [ Response   ]     [Human Agent ]
```

---

## Project Structure

```
├── agent/
│   ├── __init__.py
│   ├── agent.py          # Core SupportAgent orchestrator & decision logic
│   ├── retriever.py      # FAISS vector store manager & embedding fallback
│   ├── tools.py          # Mock database tools & order ID extraction regex
│   └── demo.py           # CLI interactive demo & automated test runs
├── data/
│   └── orders.json       # Mock order status database
├── eval/
│   ├── eval_questions.jsonl  # Evaluation test suite questions
│   ├── run_eval.py           # Automated evaluation runner
│   └── results.md            # Benchmark test results
├── kb/                   # Store policy knowledge base (16 markdown files)
│   ├── 01_return_policy.md
│   ├── 02_shipping_timelines.md
│   └── ...
├── app.py                # Streamlit web UI application
├── .env.example          # Environment variables template
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- (Optional) OpenAI API key for GPT response generation

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/varmaleena/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup** (Optional):
   Copy `.env.example` to `.env` and fill in your API key if you wish to use OpenAI LLM generation:
   ```bash
   cp .env.example .env
   ```
   *(Note: If no API key is supplied, the agent automatically runs using local sentence-transformers embeddings and deterministic template fallback responses).*

---

## Running the Application

### 1. Streamlit Web App
Launch the interactive web chat interface:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 2. CLI Demo
Run sample query tests and interactive terminal chat:
```bash
python agent/demo.py
```

### 3. Evaluation Benchmark Suite
Run the evaluation suite against the benchmark questions:
```bash
python eval/run_eval.py
```
This generates a detailed evaluation report saved to `eval/results.md`.

---

## Knowledge Base & Customization

- To add new store policies, add `.md` files to the `kb/` directory. The retriever automatically chunks and re-indexes new documents when starting up.
- To modify mock order details, edit `data/orders.json`.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
