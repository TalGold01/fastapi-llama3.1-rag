# Enterprise RAG Microservice: Air-Gapped Document Intelligence

## 📌 Executive Summary
This project is a production-ready, fully containerized Retrieval-Augmented Generation (RAG) microservice. It allows enterprise applications to securely query internal, proprietary documents using Large Language Models (LLMs) **without exposing sensitive data to external public APIs** (like OpenAI or Anthropic). 

By utilizing local model inferencing and in-memory vector indexing, this architecture ensures zero data leakage, making it ideal for defense, healthcare, and financial sectors.

## 🏗️ Architecture & Technology Stack
This microservice is built for high concurrency, low latency, and secure deployment.

* **API Gateway / Routing:** `FastAPI` (Python)
  * *Why:* Chosen for its asynchronous capabilities, native Pydantic validation, and automatic OpenAPI documentation. Perfect for high-throughput microservices.
* **Large Language Model (LLM):** `Meta Llama 3.1 (8B)` via `Ollama`
  * *Why:* The 8-Billion parameter Llama 3.1 model offers the optimal balance between high-reasoning capabilities and low VRAM requirements, allowing it to run entirely on-premises or on edge infrastructure.
* **Vector Database:** `FAISS` (Facebook AI Similarity Search)
  * *Why:* Extremely fast, locally hosted vector store that doesn't require a separate network hop (unlike hosted Pinecone), reducing latency for document retrieval.
* **AI Orchestration:** `LangChain`
  * *Why:* Standardizes the chaining of prompts, embedding models, and vector stores, making it easy to swap models if infrastructure requirements change.
* **Containerization:** `Docker`
  * *Why:* Ensures the Python environment, C++ bindings for FAISS, and routing logic are reproducible across any Kubernetes cluster or cloud VM.

## 🚀 How the RAG Pipeline Works
1. **Ingestion:** Domain-specific documents are chunked and converted into dense vector embeddings using HuggingFace embedding models.
2. **Storage:** Embeddings are indexed into the FAISS vector database.
3. **Retrieval:** When a user submits a query via the `/query` endpoint, the system converts the query to a vector and retrieves the top-K most semantically relevant document chunks.
4. **Generation:** The retrieved context is injected into a strict system prompt and sent to the local Llama 3.1 model, forcing the AI to answer *only* based on the provided documents, effectively eliminating hallucinations.

## 🛠️ Quick Start (Docker Deployment)
```bash
# 1. Clone the repository
git clone [https://github.com/TalGold01/Enterprise-RAG-Microservice.git](https://github.com/TalGold01/Enterprise-RAG-Microservice.git)
cd Enterprise-RAG-Microservice

# 2. Build the Docker Container
docker build -t local-rag-api .

# 3. Run the Microservice (Exposes port 8000)
docker run -p 8000:8000 local-rag-api
