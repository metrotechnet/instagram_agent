from fastapi import FastAPI
from pipeline import run_pipeline
import chromadb
from chromadb.utils import embedding_functions
from config import *
from openai import OpenAI

app = FastAPI(title="Instagram AI Agent")

# Chroma client pour RAG
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-large"
)
chroma_client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_DIR
))
collection = chroma_client.get_collection("instagram_transcripts")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

@app.get("/")
def home():
    return {"message": "Instagram AI Agent running!"}

@app.post("/query")
def query_agent(question: str, top_k: int = 3):
    results = collection.query(query_texts=[question], n_results=top_k)
    context = "\n".join([doc for doc in results['documents'][0]])
    prompt = f"Réponds à la question uniquement avec le contexte ci-dessous:\n{context}\n\nQuestion: {question}"

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answer": response.choices[0].message.content}

@app.post("/update")
def update_pipeline(limit: int = 5):
    run_pipeline(limit=limit)
    return {"status": "Pipeline exécuté, nouvelles vidéos indexées"}
