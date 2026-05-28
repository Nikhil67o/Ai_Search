from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from google import genai
from google.genai import types
import os
import time

# 1. FastAPI App Initialization (Yeh line miss ho gayi thi!)
app = FastAPI(title="Gemini AI Search & Chat Engine API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Client Setup
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCqJ90j8vwbHA_l3M5i4xBmaoLkA_EQxE8")
client = genai.Client(api_key=GEMINI_KEY)

# ChromaDB Setup
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="search_collection")

# Knowledge Base Sample Data
knowledge_base = [
    {"id": "1", "text": "Kubernetes (K8s) is an open-source container orchestration platform. It is used for automating deployment, scaling, and management of containerized applications."},
    {"id": "2", "text": "Docker is a platform designed to help developers build, share, and run modern applications using lightweight containers."},
    {"id": "3", "text": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints."}
]

for item in knowledge_base:
    collection.upsert(ids=[item["id"]], documents=[item["text"]])

class SearchQuery(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Gemini AI Search Engine Backend is Running smoothly!"}

@app.post("/search")
def gemini_chat_search(data: SearchQuery):
    if not data.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    context = ""
    try:
        db_results = collection.query(query_texts=[data.query], n_results=1)
        if db_results['documents'] and len(db_results['documents'][0]) > 0:
            context = db_results['documents'][0][0]
    except Exception:
        context = ""
    
    system_instruction = (
        "You are an expert DevOps and Technical AI Assistant. Provide a detailed, fully comprehensive, "
        "and complete answer to the user's query in clean English prose with proper bullet points or markdown code blocks where applicable. "
        f"Context from database: {context}"
    )
    
    # Smart Retry Framework (Exponential Backoff for 429/503 handling)
    wait_time = 2
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=data.query,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=2048,
                    temperature=0.7,
                ),
            )
            return {"results": [response.text]}
            
        except Exception as e:
            error_str = str(e).upper()
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                return {"results": ["⚠️ Google Gemini Free Tier Quota Limit reached (15-20 requests/min). Please wait 30-40 seconds and try again!"]}
            
            if attempt == 2:
                return {"results": ["Google AI Studio servers are heavily loaded right now. Please hit search again after a brief moment."]}
            
            time.sleep(wait_time)
            wait_time *= 2

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
