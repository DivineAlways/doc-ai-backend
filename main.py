from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
    raise ValueError("⚠️ Missing OpenAI API Key. Set it as an environment variable.")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI
app = FastAPI()

# Initialize ChromaDB (Vector Database)
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
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Failed to extract text: {str(e)}")


def generate_embedding(text: str):
    """Generate embeddings using OpenAI's new API."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Failed to generate embedding: {str(e)}")


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Upload a PDF file, extract text, and store embeddings in ChromaDB."""
    file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = extract_text_from_pdf(file_path)
        if not text:
            raise HTTPException(status_code=400, detail="⚠️ No extractable text found in PDF.")

        embedding = generate_embedding(text)

        # Store in ChromaDB
        document_store.add(
            ids=[str(uuid4())],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"user_id": user_id, "file_name": file.filename}]
        )

        return {"message": "✅ File uploaded and processed successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Upload failed: {str(e)}")


@app.post("/query/")
async def query_documents(query: str, user_id: str):
    """Search stored documents using embeddings and return AI-generated responses."""
    try:
        query_embedding = generate_embedding(query)

        # Retrieve relevant documents
        results = document_store.query(query_embeddings=[query_embedding], n_results=3)

        if not results["documents"]:
            return {"response": "⚠️ No matching documents found."}

        relevant_texts = " ".join([doc for doc in results["documents"][0]])

        # Generate AI response using GPT
        completion = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that answers based on uploaded documents."},
                {"role": "user", "content": query + "\n\nRelevant document excerpts: " + relevant_texts}
            ]
        )

        return {"response": completion.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Query failed: {str(e)}")


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "✅ API is running."}
