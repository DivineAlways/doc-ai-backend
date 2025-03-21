from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import shutil
from pypdf import PdfReader
import chromadb
from typing import List
from uuid import uuid4

# Load API Key from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize FastAPI
app = FastAPI()

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to restrict specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB (Vector Database)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
document_store = chroma_client.get_or_create_collection(name="documents")

# Directory to store uploaded files
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
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")


def generate_embedding(text: str):
    """Generate embeddings using OpenAI API (updated method)."""
    response = openai.embeddings.create(
        input=text, 
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Uploads a PDF file, extracts text, and stores embeddings."""
    file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    # Extract text from PDF
    text = extract_text_from_pdf(file_path)
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")

    # Generate Embeddings
    embedding = generate_embedding(text)

    # Store in ChromaDB
    document_store.add(
        ids=[str(uuid4())],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"user_id": user_id, "file_name": file.filename}]
    )
    
    return {"message": "✅ File uploaded and processed successfully!"}


@app.post("/query/")
async def query_documents(query: str, user_id: str):
    """Searches stored documents and returns AI-generated responses."""
    query_embedding = generate_embedding(query)
    results = document_store.query(query_embeddings=[query_embedding], n_results=3)

    if not results or not results["documents"]:
        return {"response": "No relevant documents found."}

    relevant_texts = " ".join([doc for doc in results["documents"][0]])

    completion = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps answer user questions based on provided documents."},
            {"role": "user", "content": query + "\n\nRelevant document excerpts: " + relevant_texts}
        ]
    )

    return {"response": completion["choices"][0]["message"]["content"]}
