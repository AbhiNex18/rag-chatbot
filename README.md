# Agentic Customer Support RAG Chatbot

An intelligent customer support chatbot built using **LangChain, FAISS, and Streamlit**. It combines Retrieval-Augmented Generation (RAG) over a Markdown knowledge base with tool execution (such as mock order lookup) and confidence-based fallback routing.

---

## Key Features

- **Grounded Knowledge Base Retrieval:** Uses local `sentence-transformers` embeddings (`all-MiniLM-L6-v2`) or OpenAI embeddings with FAISS vector similarity search.
- **Confidence Scoring and Escalation:** Calculates retrieval similarity scores and escalates queries when confidence falls below the configured threshold.
- **Tool Execution and Parameter Validation:** Detects order-tracking requests and invokes order lookup tools. Requests missing Order IDs instead of inventing parameters.
- **Out-of-Scope Detection:** Routes unrelated queries, such as general knowledge and code-generation requests, to escalation.
- **Interactive Web Interface:** Streamlit UI with decision badges, expandable reasoning logs, and preset query buttons.
- **Automated Evaluation Suite:** Includes evaluation scripts to measure query accuracy, intent routing, latency, and hallucination-related outcomes.

---

## Architecture Overview

```text
                  +----------------------+
                  |     User Query       |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |    Intent Router     |
                  +----+------+----------+
                       |      |
           +-----------+      +------------------+
           |                                     |
           v                                     v
    +--------------+                     +----------------+
    | Parameter    |                     | Vector Search  |
    | Validation   |                     | (FAISS Index)  |
    +------+-------+                     +-------+--------+
           |                                     |
      +----+-----+                               v
      |          |                      Confidence Check
    Found     Missing                   /             \
      |          |                  High             Low
      v          v                   |                |
  Execute    Request ID               v                v
  Tool       Clarification       Generate         Escalate to
                                Response          Human Agent
```

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | LLM and RAG integration |
| FAISS | Vector similarity search |
| Sentence Transformers | Local text embeddings |
| OpenAI | Optional LLM and embeddings |
| Streamlit | Web application interface |
| JSON | Mock order database |
| Markdown | Knowledge base documents |

---

## Project Structure

```text
rag-chatbot/
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── retriever.py
│   ├── tools.py
│   └── demo.py
│
├── data/
│   └── orders.json
│
├── eval/
│   ├── eval_questions.jsonl
│   ├── run_eval.py
│   └── results.md
│
├── kb/
│   ├── 01_return_policy.md
│   ├── 02_shipping_timelines.md
│   └── ...
│
├── app.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9 or later
- pip
- OpenAI API key (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/AbhiNex18/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and add the required API key if you want to use OpenAI services.

**Windows:**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Keep your API keys private. Never commit your `.env` file.

---

## Running the Application

### Streamlit Web App

```bash
streamlit run app.py
```

Open the application at:

`http://localhost:8501`

### CLI Demo

```bash
python agent/demo.py
```

### Evaluation Suite

```bash
python eval/run_eval.py
```

The evaluation script is configured to run the benchmark questions and report the results.

---

## Knowledge Base Customization

- Add new Markdown (`.md`) policy files inside the `kb/` directory.
- Modify mock order details in `data/orders.json`.
- Adjust the agent's routing and retrieval behavior in the `agent/` directory.
---

