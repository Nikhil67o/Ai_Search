from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from google import genai
from google.genai import types
import os

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
# Pro-tip: Isko chalane ke liye terminal par 'export GEMINI_API_KEY="your_key"' set karna hoga
# Agar direct code mein daalna hai toh: client = genai.Client(api_key="AIzaSy...")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
client = genai.Client(api_key=GEMINI_KEY)

# ChromaDB Setup (Lightweight local memory mode)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="search_collection")

# Knowledge Base Data
knowledge_base = [
    {"id": "1", "text": "Kubernetes (K8s) ek open-source container orchestration platform hai. Yeh automated deployment, scaling, aur containerized applications ke management ke liye use hota hai. Iske paas self-healing aur load balancing jaise features hote hain."},
    {"id": "2", "text": "Docker ek platform hai jo applications ko lightweight containers mein pack, ship, aur run karne ke kaam aata hai. Isse code har environment mein bina kisi mismatch ke smoothly chalta hai."},
    {"id": "3", "text": "FastAPI ek extremely fast, modern web framework hai Python 3.8+ ke liye, jo standard Python type hints par based hai APIs build karne ke liye."}
]

# Startup par data insert karein
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
    
    # 1. Vector DB se nearest match uthao (Context Extraction)
    db_results = collection.query(
        query_texts=[data.query],
        n_results=1
    )
    
    context = ""
    if db_results['documents'] and len(db_results['documents'][0]) > 0:
        context = db_results['documents'][0][0]
    
    # 2. System Instruction aur User Query ko combine karke Gemini ko bhejna
    system_instruction = (
        "Aap ek helpful AI assistant hain. Diye gaye Context ke basis par user ke sawaal ka detailed aur "
        "proper answer generate kijiye. Agar information context mein na ho, toh apne general knowledge "
        f"se best response dijiye.\n\nContext: {context}"
    )
    
    try:
        # Hum use kar rahe hain gemini-2.5-flash (jo ki super fast aur advanced model hai)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=data.query,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=500,
                temperature=0.7,
            ),
        )
        
        gemini_answer = response.text
        return {"results": [gemini_answer]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")