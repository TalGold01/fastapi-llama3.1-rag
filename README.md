# Enterprise RAG Microservice: Air-Gapped Document Intelligence

## 📌 Executive Summary
This project is a production-ready, fully containerized Retrieval-Augmented Generation (RAG) microservice. It allows enterprise applications to securely query internal, proprietary documents using Large Language Models (LLMs) without exposing sensitive data to external public APIs (like OpenAI or Anthropic).

By utilizing local model inferencing, decoupled frontend/backend services, and in-memory vector indexing, this architecture ensures zero data leakage. Recent architectural overhauls have introduced asynchronous token streaming for zero-latency UI rendering and a programmatic benchmarking suite to validate enterprise performance metrics.

## 🏗️ Architecture & Technology Stack
This project utilizes a modern microservice architecture, separating the inference engine, routing logic, and presentation layer.

* **API Gateway / Backend:** `FastAPI` (Python)
  * *Why:* Chosen for its asynchronous capabilities. Upgraded to utilize `StreamingResponse` to deliver real-time token streaming to the client.
* **Frontend / Presentation Layer:** `Streamlit` (Python)
  * *Why:* Enables rapid deployment of an interactive chat interface that consumes chunked HTTP responses for a ChatGPT-like typing effect.
* **Large Language Model (LLM):** `Meta Llama 3.1 (8B)` via `Ollama`
  * *Why:* The optimal balance between high-reasoning capabilities and low VRAM requirements, running entirely on-premises.
* **Vector Database:** `FAISS` (Facebook AI Similarity Search)
  * *Why:* Extremely fast, locally hosted vector store. Now supports dynamic runtime ingestion of documents.
* **AI Orchestration:** `LangChain (LCEL)`
  * *Why:* Utilizes modern LangChain Expression Language (LCEL) with asynchronous generators (`astream`) for optimized Time To First Token (TTFT).
* **Containerization & Security:** `Docker` & `Docker Compose`
  * *Why:* The backend is secured inside an isolated `internal: true` Docker bridge network, enforcing strict air-gapped constraints and avoiding the vulnerabilities of host network modes.

## 🚀 How the RAG Pipeline Works
1. **Dynamic Ingestion:** Domain-specific documents (.pdf, .txt) are uploaded at runtime via the `/upload` API, chunked using a `RecursiveCharacterTextSplitter`, and converted into dense vector embeddings using HuggingFace embedding models.
2. **Storage:** These embeddings are dynamically indexed into the active, in-memory `FAISS` vector database without requiring a system restart.
3. **Retrieval:** When a user submits a query via the Streamlit UI, the FastAPI backend converts the query to a vector and retrieves the top-K most semantically relevant document chunks.
4. **Asynchronous Generation:** The retrieved context is injected into a strict system prompt and sent to the local Llama 3.1 model. The response is streamed back token-by-token using `astream()`, effectively eliminating hallucinations while providing zero-latency feedback.

---

## 🛠️ Quick Start (Local Deployment)

This application requires three separate services to run simultaneously: the AI Engine, the API Backend, and the UI Frontend.

### Prerequisites
* Docker and Docker Compose installed.
* Ollama installed natively for local LLM inference.
* Python 3.10+ installed natively.

### Step 1: Start the Inference Engine (The Brain)
Open your terminal and start the local Ollama daemon. You must bind it to all host interfaces so the secure Docker bridge can reach it.

    ollama pull llama3.1
    OLLAMA_HOST=0.0.0.0 ollama serve

### Step 2: Start the FastAPI Backend (The Engine)
Open a second terminal window, clone the repository, and spin up the Docker container using Compose. This builds the HuggingFace embeddings directly into the image to preserve the air-gap.

    git clone https://github.com/TalGold01/fastapi-llama3.1-rag.git
    cd fastapi-llama3.1-rag
    docker-compose up --build -d

*The backend API is now actively listening and securely mapped to http://localhost:8000.*

### Step 3: Start the Streamlit UI (The Face)
Open a third terminal window, initialize a virtual environment, and launch the user interface:

    cd fastapi-llama3.1-rag
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    streamlit run ui.py

*Navigate to http://localhost:8501 in your browser to interact with the system!*

---

## 📡 Core API Endpoints
The microservice exposes the following programmatic endpoints:

* `GET /health`: Returns the health status of the Kubernetes/Load Balancer pod and verifies the AI pipeline is active.
* `POST /upload`: Accepts `.pdf` or `.txt` multipart form uploads, automatically chunks the text, and ingests it into the live FAISS vector database.
* `POST /query`: Accepts a JSON payload `{"question": "your string here"}` and returns an HTTP chunked stream of the generated answer, appending cited sources at the end.

---

## 📊 Observability & Benchmarking

A dedicated benchmarking suite (`benchmark.py`) measures end-to-end Time To First Token (TTFT) across P50/P95/P99 percentiles.

> **Note:** TTFT is hardware-dependent. CPU-only inference (laptop) produces multi-second latency due to the 8B parameter model size. Production deployment on GPU hardware (e.g., AWS g4dn with NVIDIA T4) reduces TTFT to the sub-500ms range. Run `benchmark.py` to measure on your own hardware.

### Running the Benchmark
With the backend running, execute the benchmarking suite to stress-test the `/query` endpoint:

    python benchmark.py
