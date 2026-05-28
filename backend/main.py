from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI(title="AI Search Engine API")

# Frontend se connect karne ke liye CORS allow karein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB Client setup (In-memory for simplicity)
chroma_client = chromadb.Client()
# Sentence Transformers use karenge text ko vectors/embeddings mein badalne ke liye
model_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="search_collection", embedding_function=model_fn)

# Kuch sample data insert karte hain jo AI search test karne ke kaam aayega
sample_data = [
    {"id": "1", "text": "Kubernetes ek open-source container orchestration tool hai jo deployments ko automate karta hai."},
    {"id": "2", "text": "Docker containers ko pack aur run karne ke liye ek lightweight platform hai."},
    {"id": "3", "text": "FastAPI Python ka ek modern aur fast web framework hai APIs banane ke liye."},
    {"id": "4", "text": "AI Search Engine context aur meaning ko samajhta hai, sirf keywords ko nahi."}
]

# Startup par hi data insert kar dete hain
for item in sample_data:
    collection.upsert(ids=[item["id"]], documents=[item["text"]])

class SearchQuery(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "AI Search Engine Backend is Running!"}

@app.post("/search")
def search(data: SearchQuery):
    if not data.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Vector DB se query match karte hain (Top 2 results)
    results = collection.query(
        query_texts=[data.query],
        n_results=2
    )
    
    # Results ko clean format mein bhejte hain
    formatted_results = []
    if results['documents']:
        for doc in results['documents'][0]:
            formatted_results.append(doc)
            
    return {"results": formatted_results}