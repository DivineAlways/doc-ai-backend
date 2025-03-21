from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import shutil

# Initialize FastAPI
app = FastAPI()

# Enable CORS (for frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Gemini API Key (replace with your key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in your environment
genai.configure(api_key=GEMINI_API_KEY)

# Store uploaded files temporarily
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Handles file upload."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"message": "Upload successful!", "file_path": file_path, "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/query/")
async def query_ai(query: str, user_id: str):
    """Handles AI query using Google Gemini."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(query)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "API is running!"}
