"""
Haley Chat Service - Simplified chat with ElevenLabs voice
Deploy to Cloud Run for a single URL endpoint
"""

import os
import logging
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, List
import httpx
from io import BytesIO
import uuid
import json
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config from environment
OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://localhost:18789")
OPENCLAW_TOKEN = os.getenv("OPENCLAW_TOKEN", "cloud-run-bundled-token")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")

# GCS config
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "haley_chat")
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)

# Auth config
AUTH_PASSWORD = os.getenv("CHAT_PASSWORD", "default-password")

# Session storage
sessions: Dict[str, List[dict]] = {}

# Create app
app = FastAPI(title="Haley Chat")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================================
# Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class SpeakRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "cgSgspJ2msm6clMCkdW9"


# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat interface"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "openclaw_url": OPENCLAW_URL}


@app.post("/api/auth")
async def auth(request: Request):
    """Password gate authentication"""
    data = await request.json()
    if data.get("password") == AUTH_PASSWORD:
        return {"token": "session-token", "valid": True}
    return JSONResponse(status_code=401, content={"valid": False})


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload file to GCS"""
    blob = bucket.blob(f"uploads/{file.filename}")
    blob.upload_from_file(file.file)
    return {"url": blob.public_url, "filename": file.filename}


@app.get("/api/files")
async def list_files():
    """List files in GCS bucket"""
    blobs = bucket.list_blobs(prefix="uploads/")
    return {"files": [{"name": b.name, "size": b.size} for b in blobs]}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send message to OpenClaw and get response"""
    try:
        logger.info(f"Chat message: {request.message[:50]}...")
        
        # Generate or reuse session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Call OpenClaw's OpenAI-compatible endpoint
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Content-Type": "application/json"
            }
            if OPENCLAW_TOKEN:
                headers["Authorization"] = f"Bearer {OPENCLAW_TOKEN}"
            
            response = await client.post(
                f"{OPENCLAW_URL}/v1/chat/completions",
                headers=headers,
                json={
                    "model": "openclaw:main",
                    "messages": [{"role": "user", "content": request.message}],
                    "stream": False,
                    "user": session_id  # For session persistence
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "response": assistant_message,
                    "status": "success",
                    "session_id": session_id
                }
            else:
                logger.error(f"OpenClaw error: {response.status_code} - {response.text}")
                return {
                    "response": f"OpenClaw returned error {response.status_code}. Please try again.",
                    "status": "error",
                    "session_id": session_id
                }
                
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": "Something went wrong. Try again?", "status": "error"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            logger.info(f"WebSocket message: {user_message[:50]}...")
            
            # Echo back for now - replace with actual OpenClaw integration
            response = {
                "type": "message",
                "content": f"Echo: {user_message}",
                "session_id": session_id
            }
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe audio using Whisper"""
    try:
        if not OPENAI_API_KEY:
            return {"error": "OpenAI API key not configured"}
        
        # Read audio file
        audio_content = await audio.read()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {
                "file": ("audio.webm", BytesIO(audio_content), "audio/webm"),
                "model": (None, "whisper-1")
            }
            
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files=files
            )
            
            if response.status_code == 200:
                data = response.json()
                return {"text": data.get("text", ""), "status": "success"}
            else:
                logger.error(f"Whisper error: {response.text}")
                return {"error": "Transcription failed"}
                
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        return {"error": str(e)}


@app.post("/api/speak")
async def speak(request: SpeakRequest):
    """Text to speech using ElevenLabs"""
    try:
        if not ELEVENLABS_API_KEY:
            return {"error": "ElevenLabs API key not configured"}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{request.voice_id}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "text": request.text,
                    "model_id": "eleven_multilingual_v2"
                }
            )
            
            if response.status_code == 200:
                return StreamingResponse(
                    BytesIO(response.content),
                    media_type="audio/mpeg"
                )
            else:
                logger.error(f"TTS error: {response.text}")
                return {"error": "TTS failed"}
                
    except Exception as e:
        logger.error(f"Speak error: {e}")
        return {"error": str(e)}


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
