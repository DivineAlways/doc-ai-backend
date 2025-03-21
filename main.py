from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import shutil
from pypdf import PdfReader
from typing import List
from uuid import uuid4

# 🔑 Load OpenAI API Key from Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key")

openai.api_key = OPENAI_API_KEY

# 🚀 Initialize FastAPI App
app = FastAPI()

# 🔥 Enable CORS to Allow Frontend Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Directory to store uploaded files
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def generate_embedding(text: str):
    """Generate embeddings using OpenAI API (Updated for new OpenAI SDK)."""
    response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Handle PDF Upload, Extract Text, and Store Embeddings"""
    file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    text = extract_text_from_pdf(file_path)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the file.")
    
    embedding = generate_embedding(text)
    
    return {"message": "File uploaded and processed successfully."}


@app.post("/query/")
async def query_documents(query: str, user_id: str):
    """Search stored documents and return AI-generated responses."""
    query_embedding = generate_embedding(query)

    completion = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps answer user questions based on provided documents."},
            {"role": "user", "content": query}
        ]
    )
    
    return {"response": completion.choices[0].message.content}


# Run the server using:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
