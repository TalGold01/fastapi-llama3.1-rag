import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- App Initialization ---
app = FastAPI(
    title="Enterprise RAG API",
    description="Secure, local Document Retrieval and LLM Generation via Llama 3.1",
    version="1.0.0"
)

# --- Global State & Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "llama3.1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Pydantic Models for Data Validation ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: list[str]

# --- AI Pipeline Initialization ---
def initialize_rag_pipeline():
    try:
        # 1. Initialize Local LLM & Embeddings
        llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # 2. Initialize FAISS Vector Store
        sample_texts = [
            "The main server rack requires 220V power.", 
            "Active Directory policies dictate 90-day password rotations."
        ]
        vector_db = FAISS.from_texts(sample_texts, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        
        # 3. Strict System Prompt
        prompt_template = """
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Do NOT make up an answer.
        
        Context: {context}
        Question: {question}
        
        Helpful Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # 4. Helper to format retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 5. Build the Modern LCEL Pipeline (Replacing legacy RetrievalQA)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever
        
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        return None, None

# Load the chain into memory on startup
rag_chain, retriever = initialize_rag_pipeline()

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Kubernetes / Load Balancer health check endpoint."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="AI Pipeline not initialized")
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest):
    """
    Retrieves context using FAISS, formats it via LCEL, 
    and generates an answer securely.
    """
    if not rag_chain or not retriever:
        raise HTTPException(status_code=500, detail="AI Service is currently unavailable")
    
    try:
        # 1. Fetch sources directly for citations
        docs = retriever.invoke(req.question)
        sources = [doc.page_content for doc in docs]
        
        # 2. Generate the LLM answer using the LCEL chain
        answer = rag_chain.invoke(req.question)
        
        return QueryResponse(
            answer=answer,
            source_documents=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)