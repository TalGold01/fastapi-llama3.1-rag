# Enterprise RAG Microservice: Air-Gapped Document Intelligence

## 📌 Executive Summary
This project is a production-ready, fully containerized Retrieval-Augmented Generation (RAG) microservice. It allows enterprise applications to securely query internal, proprietary documents using Large Language Models (LLMs) without exposing sensitive data to external public APIs (like OpenAI or Anthropic). 

By utilizing local model inferencing, decoupled frontend/backend services, and in-memory vector indexing, this architecture ensures zero data leakage, making it ideal for defense, healthcare, and financial sectors.

## 🏗️ Architecture & Technology Stack
This project utilizes a modern microservice architecture, separating the inference engine, routing logic, and presentation layer.

* **API Gateway / Backend:** `FastAPI` (Python)
  * *Why:* Chosen for its asynchronous capabilities, native Pydantic validation, and automatic OpenAPI documentation. Perfect for high-throughput microservices.
* **Frontend / Presentation Layer:** `Streamlit` (Python)
  * *Why:* Enables rapid deployment of an interactive, decoupled chat interface that communicates with the backend API via standard HTTP requests.
* **Large Language Model (LLM):** `Meta Llama 3.1 (8B)` via `Ollama`
  * *Why:* The 8-Billion parameter Llama 3.1 model offers the optimal balance between high-reasoning capabilities and low VRAM requirements, allowing it to run entirely on-premises.
* **Vector Database:** `FAISS` (Facebook AI Similarity Search)
  * *Why:* Extremely fast, locally hosted vector store that doesn't require a separate network hop (unlike cloud-hosted alternatives), minimizing latency for document retrieval.
* **AI Orchestration:** `LangChain (LCEL)`
  * *Why:* Utilizes modern LangChain Expression Language (LCEL) for optimized, readable, and streaming-ready inference routing, abandoning legacy chain wrappers.
* **Containerization:** `Docker`
  * *Why:* Ensures the Python environment, C++ bindings for FAISS, and routing logic are reproducible across any Kubernetes cluster or cloud VM.

## 🚀 How the RAG Pipeline Works
1. **Ingestion:** Domain-specific documents are chunked and converted into dense vector embeddings using HuggingFace embedding models.
2. **Storage:** Embeddings are indexed into the FAISS vector database.
3. **Retrieval:** When a user submits a query via the Streamlit UI, the FastAPI backend converts the query to a vector and retrieves the top-K most semantically relevant document chunks.
4. **Generation:** The retrieved context is injected into a strict system prompt and sent to the local Llama 3.1 model, forcing the AI to answer *only* based on the provided documents, effectively eliminating hallucinations.

---

## 🛠️ Quick Start (Local Deployment)

This application requires three separate services to run simultaneously: the AI Engine, the API Backend, and the UI Frontend. 

### Prerequisites
* Docker installed.
* Ollama installed for local LLM inference.
* Python 3.10+ installed natively.

### Step 1: Start the Inference Engine (The Brain)
Open your terminal and start the local Ollama daemon. You will need to pull the Llama 3.1 model if this is your first time running it.

    ollama pull llama3.1
    ollama serve

### Step 2: Start the FastAPI Backend (The Engine)
Open a second terminal window, clone the repository, and spin up the Docker container. 
(Note: the --network host flag is required for the Docker container to securely communicate with your local Ollama daemon).

    git clone https://github.com/TalGold01/fastapi-llama3.1-rag.git
    cd fastapi-llama3.1-rag
    docker build -t fastapi-llama3.1-rag .
    docker run --network host fastapi-llama3.1-rag

*The backend API is now actively listening on http://localhost:8000.*

### Step 3: Start the Streamlit UI (The Face)
Open a third terminal window, initialize a virtual environment to avoid system conflicts, and launch the user interface:

    cd fastapi-llama3.1-rag
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    streamlit run ui.py

*Navigate to http://localhost:8501 in your browser to interact with the secure document intelligence hub!*

---

## 📡 Core API Endpoints
If you prefer to bypass the UI and communicate with the microservice directly, you can hit the FastAPI endpoints:

* `GET /health`: Returns the health status of the Kubernetes/Load Balancer pod and verifies the LCEL pipeline is loaded into memory.
* `POST /query`: Accepts a JSON payload `{"question": "your string here"}` and returns the generated answer alongside the specific document chunks used for context.
* `GET /docs`: Auto-generated Swagger UI for visual API testing.