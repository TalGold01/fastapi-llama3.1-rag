import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# --- App Initialization ---
app = FastAPI(
    title="Enterprise RAG API",
    description="Secure, local Document Retrieval and LLM Generation via Llama 3.1",
    version="1.1.0"
)

# --- Global State & Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "llama3.1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Pydantic Models for Data Validation ---
class QueryRequest(BaseModel):
    question: str

# --- AI Pipeline Initialization ---
def initialize_rag_pipeline():
    try:
        # 1. Initialize Local LLM & Embeddings
        llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # 2. Seed Document (FAISS requires at least one document to initialize)
        raw_documents = [
            "System Initialization Document: This enterprise RAG system is currently active, secure, and awaiting enterprise document ingestion."
        ]
        
        # 3. ADVANCED CHUNKING STRATEGY
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.create_documents(raw_documents)
        
        # 4. Initialize FAISS Vector Store
        vector_db = FAISS.from_documents(split_docs, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        # 5. Strict System Prompt
        prompt_template = """
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Do NOT make up an answer.
        
        Context: {context}
        Question: {question}
        
        Helpful Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # 6. Helper to format retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 7. Build the Modern LCEL Pipeline
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )
        
        # We now return the vector_db and text_splitter so the /upload route can access them globally
        return rag_chain, retriever, vector_db, text_splitter
        
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        return None, None, None, None

# Load the chain and database into memory on startup
rag_chain, retriever, vector_db, text_splitter = initialize_rag_pipeline()


# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Kubernetes / Load Balancer health check endpoint."""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="AI Pipeline not initialized")
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests PDF or TXT files, chunks them, and adds them to the live FAISS vector database.
    """
    if not vector_db:
        raise HTTPException(status_code=500, detail="Vector Database not initialized")
    
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in ['.pdf', '.txt']:
            raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Load the document based on extension
        if suffix == '.pdf':
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
            
        documents = loader.load()
        
        # Chunk the documents and add to FAISS
        split_docs = text_splitter.split_documents(documents)
        vector_db.add_documents(split_docs)
        
        # Clean up temp file
        os.remove(tmp_path)
        
        return {"message": f"Successfully ingested {file.filename}. Added {len(split_docs)} vector chunks to the database."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")


@app.post("/query")
async def process_query_stream(req: QueryRequest):
    """
    Streams the LLM generation back to the client token-by-token.
    Appends the retrieved source citations at the end of the stream.
    """
    if not rag_chain or not retriever:
        raise HTTPException(status_code=500, detail="AI Service is currently unavailable")
    
    async def generate_response():
        # 1. Stream the LLM response natively
        async for chunk in rag_chain.astream(req.question):
            yield chunk
            
        # 2. Append the exact sources used to the end of the stream
        yield "\n\n--- CITED SOURCES ---\n"
        docs = retriever.invoke(req.question)
        if not docs:
            yield "No external documents retrieved."
        else:
            for i, doc in enumerate(docs):
                # Clean up newlines for cleaner output and truncate for readability
                clean_content = doc.page_content.replace('\n', ' ')[:250]
                yield f"[{i+1}] {clean_content}...\n"

    # Return the generator as a StreamingResponse
    return StreamingResponse(generate_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)