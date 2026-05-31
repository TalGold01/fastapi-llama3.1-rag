import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- App Initialization ---
app = FastAPI(
    title="Enterprise RAG API",
    description="Secure, local Document Retrieval and LLM Generation via Llama 3.1",
    version="1.0.0"
)

# --- Global State & Configuration ---
# In production, these would be loaded via environment variables / secure vaults
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
        # 1. Initialize the Local LLM (Llama 3.1)
        llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
        
        # 2. Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # 3. Load or Create FAISS Vector Store
        # (For this lab, we mock a basic loaded DB. In production, this loads from a persistent volume)
        sample_texts = [
            "The main server rack requires 220V power.", 
            "Active Directory policies dictate 90-day password rotations."
        ]
        vector_db = FAISS.from_texts(sample_texts, embeddings)
        
        # 4. Strict System Prompt to prevent Hallucinations
        prompt_template = """
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Do NOT make up an answer.
        
        Context: {context}
        Question: {question}
        
        Helpful Answer:"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # 5. Build the Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain
        
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        return None

# Load the chain into memory on startup
qa_chain = initialize_rag_pipeline()

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Kubernetes / Load Balancer health check endpoint."""
    if not qa_chain:
        raise HTTPException(status_code=503, detail="AI Pipeline not initialized")
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest):
    """
    Takes a user question, searches the FAISS database for context, 
    and generates a secure answer using the local LLM.
    """
    if not qa_chain:
        raise HTTPException(status_code=500, detail="AI Service is currently unavailable")
    
    try:
        # Execute the RAG chain
        result = qa_chain.invoke({"query": req.question})
        
        # Extract metadata for citation
        sources = [doc.page_content for doc in result.get("source_documents", [])]
        
        return QueryResponse(
            answer=result["result"],
            source_documents=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)