import os
import shutil
import openai
import chromadb
import faiss
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List
from uuid import uuid4

# Load environment variables (for OpenAI API key)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Directory for storing uploaded files
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize ChromaDB for vector storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
document_store = chroma_client.get_or_create_collection(name="documents")

# Load Sentence Transformers Model (for local embeddings)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def generate_embedding_local(text: str):
    """Generate embeddings using Sentence Transformers (FAISS)."""
    return embed_model.encode(text).tolist()


def generate_embedding_openai(text: str):
    """Generate embeddings using GPT-4-Turbo instead of OpenAI Embedding API."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is missing!")

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Generate a semantic embedding vector for the given text."},
            {"role": "user", "content": text}
        ],
        response_format="json"
    )
    
    # Convert response text to a simple vector representation
    embedding_vector = [ord(char) / 255.0 for char in response["choices"][0]["message"]["content"][:512]]
    
    return embedding_vector


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...), use_openai: bool = Form(False)):
    """Upload a PDF, extract text, and store embeddings."""
    file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    text = extract_text_from_pdf(file_path)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the file.")

    # Choose embedding method
    if use_openai:
        embedding = generate_embedding_openai(text)
    else:
        embedding = generate_embedding_local(text)

    # Store in ChromaDB
    document_store.add(
        ids=[str(uuid4())],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"user_id": user_id, "file_name": file.filename}]
    )

    return {"message": "File uploaded and processed successfully."}


@app.post("/query/")
async def query_documents(query: str, user_id: str, use_openai: bool = False):
    """Query stored documents and return AI-generated responses."""
    query_embedding = generate_embedding_openai(query) if use_openai else generate_embedding_local(query)

    results = document_store.query(query_embeddings=[query_embedding], n_results=3)

    if not results["documents"]:
        return {"response": "No matching documents found."}

    relevant_texts = " ".join(results["documents"][0])

    completion = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant helping users search through documents."},
            {"role": "user", "content": query + "\n\nRelevant document excerpts: " + relevant_texts}
        ]
    )

    return {"response": completion["choices"][0]["message"]["content"]}
