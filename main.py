# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import pymongo
import uuid
from transformers import pipeline  # Using Hugging Face
import os
import aiofiles

app = FastAPI(title="RAG CSV Analyser")

# MongoDB Connection
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["csv_analyser"]
files_collection = db["files"]

# LLM Setup (using Hugging Face's distilbert for simplicity)
llm = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Models
class QueryRequest(BaseModel):
    file_id: str
    query: str

class FileResponse(BaseModel):
    file_id: str
    file_name: str

# Helper Functions
async def process_csv(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path)
        # Convert CSV to text for RAG
        content = df.to_string()
        return {"content ": content, "metadata": {"rows": len(df), "columns": list(df.columns)}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

# Endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(None), file_path: str = None):
    if not file and not file_path:
        raise HTTPException(status_code=400, detail="Provide either a file or file path")
    
    file_id = str(uuid.uuid4())
    
    try:
        # Handle direct upload
        if file:
            file_name = file.filename
            file_content = await file.read()
            async with aiofiles.open(f"temp_{file_id}.csv", "wb") as f:
                await f.write(file_content)
            processed_data = await process_csv(f"temp_{file_id}.csv")
            os.remove(f"temp_{file_id}.csv")
        
        # Handle file path
        elif file_path:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")
            file_name = os.path.basename(file_path)
            processed_data = await process_csv(file_path)
        
        # Store in MongoDB
        files_collection.insert_one({
            "file_id": file_id,
            "file_name": file_name,
            "document": processed_data["content"],
            "metadata": processed_data["metadata"]
        })
        
        return {"file_id": file_id, "message": "Upload successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")

@app.get("/files", response_model=dict[str, List[FileResponse]])
async def list_files():
    try:
        files = files_collection.find({}, {"file_id": 1, "file_name": 1, "_id": 0})
        return {"files": [FileResponse(**f) for f in files]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.post("/query")
async def query_file(request: QueryRequest):
    file = files_collection.find_one({"file_id": request.file_id})
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Simple RAG implementation
        response = llm({
            "question": request.query,
            "context": file["document"]
        })
        return {"response": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.delete("/file/{file_id}")
async def delete_file(file_id: str):
    result = files_collection.delete_one({"file_id": file_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="File not found")
    return {"message": "File deleted successfully"}

# Run the app: uvicorn main:app --reload