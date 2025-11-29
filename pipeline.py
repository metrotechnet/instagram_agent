from instagrapi import Client
from moviepy.editor import VideoFileClip
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from config import *
import os
from tqdm import tqdm

# Setup folders
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Clients
client_openai = OpenAI(api_key=OPENAI_API_KEY)

ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-large"
)
chroma_client = chromadb.Client(chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_DIR
))
collection = chroma_client.get_or_create_collection(
    name="instagram_transcripts", embedding_function=ef
)

# Instagram login
cl = Client()
cl.login(INSTAGRAM_USER, INSTAGRAM_PASS)

def run_pipeline(limit=10):
    medias = cl.user_medias(cl.user_id_from_username(TARGET_ACCOUNT), limit)

    for m in tqdm(medias, desc="Processing videos"):
        if m.media_type != 2:  # vidéo
            continue

        video_path = cl.video_download(m.pk, VIDEO_DIR)
        audio_path = video_path.replace(".mp4", ".mp3")
        transcript_path = os.path.join(TRANSCRIPTS_DIR, os.path.basename(video_path).replace(".mp4", ".txt"))

        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)

        with open(audio_path, "rb") as f:
            transcript = client_openai.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            )

        text = transcript.text
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Chunk & push to Chroma
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": os.path.basename(video_path), "chunk": i}],
                ids=[f"{m.pk}_chunk_{i}"]
            )

    chroma_client.persist()
    print("✅ Pipeline terminé")
