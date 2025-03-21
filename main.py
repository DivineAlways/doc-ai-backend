from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil

app = FastAPI()

# Allow frontend requests (change to your frontend URL)
origins = [
    "http://localhost:3000",
    "https://your-vercel-app.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for simplicity
db = {}

# Request model for querying
class QueryRequest(BaseModel):
    query: str
    user_id: str

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Handles file upload and saves it locally."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Store file metadata
    db[user_id] = file_path
    return {"message": "File uploaded successfully", "file_name": file.filename}

@app.post("/query/")
async def query_data(data: QueryRequest):
    """Processes queries related to uploaded PDFs"""
    if not data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if data.user_id not in db:
        raise HTTPException(status_code=404, detail="No file uploaded for this user")

    try:
        # Simulating a response (Replace this with Gemini API or LLM)
        response = f"AI Response for: {data.query}"
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
