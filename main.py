from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import fitz  # PyMuPDF
import sqlite3
import os

app = FastAPI()

DB_FILE = "pdf_data.db"

# Ensure database exists
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Upload endpoint
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Read and extract text from PDF
        pdf_doc = fitz.open(stream=file.file.read(), filetype="pdf")
        extracted_text = "\n".join([page.get_text() for page in pdf_doc])
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF is empty or cannot be read.")

        # Store in database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO pdfs (filename, content) VALUES (?, ?)", (file.filename, extracted_text))
        conn.commit()
        conn.close()

        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Query endpoint
class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_text(request: QueryRequest):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM pdfs ORDER BY id DESC LIMIT 1")  # Get the latest uploaded file
        result = cursor.fetchone()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="No PDFs found.")

        # Simple search logic (improve with NLP later)
        extracted_text = result[0]
        matched_text = [line for line in extracted_text.split("\n") if request.query.lower() in line.lower()]

        return {"query": request.query, "matched_text": matched_text[:5]}  # Return first 5 matches

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# List uploaded PDFs
@app.get("/list_pdfs/")
async def list_pdfs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM pdfs")
    files = cursor.fetchall()
    conn.close()

    return {"uploaded_pdfs": [f[0] for f in files]}
