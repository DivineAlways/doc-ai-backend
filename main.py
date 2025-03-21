from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import shutil
from pypdf import PdfReader
from uuid import uuid4

# ✅ Load Google Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-google-api-key-here")
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Initialize FastAPI
app = FastAPI()

# 🔥 Fix CORS Issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Upload a PDF file and extract text."""
    file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_path)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the file.")

    return {"message": "File uploaded and processed successfully.", "text": text}


@app.post("/query/")
async def query_documents(query: str, user_id: str):
    """Send query to Gemini API and return a response."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content([query])
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Run with:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
