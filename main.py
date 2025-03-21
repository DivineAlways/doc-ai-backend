from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import os
import shutil
from uuid import uuid4
from openai import OpenAI
from pypdf import PdfReader
from typing import List

# Load API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific frontend domains for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
document_store = chroma_client.get_or_create_collection(name="documents")

# Directory to store uploaded files
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def generate_embedding(text: str):
    """Generate embeddings using OpenAI API (updated for openai>=1.0.0)."""
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Upload a PDF, extract text, and store embeddings."""
    file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_path)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the file.")

    embedding = generate_embedding(text)

    document_store.add(
        ids=[str(uuid4())],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"user_id": user_id, "file_name": file.filename}]
    )

    return {"message": "File uploaded and processed successfully."}

@app.post("/query/")
async def query_documents(query: str, user_id: str):
    """Search stored documents and return AI-generated responses."""
    query_embedding = generate_embedding(query)
    results = document_store.query(query_embeddings=[query_embedding], n_results=3)

    if not results["documents"]:
        return {"response": "No relevant documents found."}

    relevant_texts = " ".join([doc for doc in results["documents"][0]])

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps answer user questions based on provided documents."},
            {"role": "user", "content": query + "\n\nRelevant document excerpts: " + relevant_texts}
        ]
    )

    return {"response": completion.choices[0].message.content}
