import os
import shutil
import google.generativeai as genai
import chromadb
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from uuid import uuid4
from pypdf import PdfReader
from fastapi.middleware.cors import CORSMiddleware

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing Google Gemini API Key")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
document_store = chroma_client.get_or_create_collection(name="documents")

# Upload Directory
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def generate_embedding(text: str):
    """Generate embeddings using Gemini API."""
    try:
        model = genai.GenerativeModel("embedding-001")
        response = model.embed_content(text)
        return response['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Uploads a PDF file, extracts text, generates embeddings, and stores in ChromaDB."""
    file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_path)
    if not text:
        raise HTTPException(status_code=400, detail="No text found in the uploaded PDF.")

    embedding = generate_embedding(text)

    document_store.add(
        ids=[str(uuid4())],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"user_id": user_id, "file_name": file.filename}]
    )

    return {"message": "File uploaded and processed successfully."}

class QueryRequest(BaseModel):
    query: str
    user_id: str

@app.post("/query/")
async def query_documents(request: QueryRequest):
    """Search stored documents and return AI-generated responses."""
    query_embedding = generate_embedding(request.query)
    results = document_store.query(query_embeddings=[query_embedding], n_results=3)

    if not results["documents"]:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    relevant_texts = " ".join([doc for doc in results["documents"][0]])

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(request.query + "\n\nRelevant excerpts: " + relevant_texts)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")
