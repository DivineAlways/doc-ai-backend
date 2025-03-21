from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import openai
import os
import shutil
from pypdf import PdfReader
from typing import List
from uuid import uuid4

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key")

openai.api_key = OPENAI_API_KEY

# Initialize FastAPI
app = FastAPI()

# Enable CORS for Frontend Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend requests
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
    """Extract text from a PDF file safely."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not text:
            raise ValueError("Could not extract text from the PDF.")
        return text
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {e}")


def generate_embedding(text: str):
    """Generate embeddings using OpenAI API."""
    try:
        response = openai.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Upload PDF, extract text, and store embeddings."""
    try:
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext != "pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = extract_text_from_pdf(file_path)
        embedding = generate_embedding(text)

        document_store.add(
            ids=[str(uuid4())],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"user_id": user_id, "file_name": file.filename}],
        )

        return {"message": "File uploaded and processed successfully."}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")


@app.post("/query/")
async def query_documents(query: str, user_id: str):
    """Search stored documents and return AI-generated responses."""
    try:
        query_embedding = generate_embedding(query)
        results = document_store.query(query_embeddings=[query_embedding], n_results=3)

        relevant_texts = " ".join([doc for doc in results["documents"][0]])

        completion = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that answers questions using stored documents.",
                },
                {"role": "user", "content": query + "\n\nRelevant document excerpts: " + relevant_texts},
            ],
        )

        return {"response": completion["choices"][0]["message"]["content"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {e}")
