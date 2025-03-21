from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import fitz  # PyMuPDF for PDF text extraction

app = FastAPI()

# Allow frontend requests
origins = ["http://localhost:3000", "https://your-vercel-app.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for simplicity
db = {}

class QueryRequest(BaseModel):
    query: str
    user_id: str

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text if text.strip() else "No readable text found in PDF."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Handles file upload and extracts text."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text and store it in-memory
    pdf_text = extract_text_from_pdf(file_path)
    db[user_id] = {"file": file_path, "content": pdf_text}

    return {"message": "File uploaded successfully", "file_name": file.filename}

@app.post("/query/")
async def query_data(data: QueryRequest):
    """Processes queries related to uploaded PDFs."""
    if not data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if data.user_id not in db:
        raise HTTPException(status_code=404, detail="No file uploaded for this user")

    try:
        pdf_content = db[data.user_id]["content"]
        response = f"Query: {data.query}\n\nExtracted PDF Content: {pdf_content[:1000]}..."  # Show first 1000 chars
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
