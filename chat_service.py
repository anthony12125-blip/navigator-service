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

# System prompt - enforces English
SYSTEM_PROMPT = (
    "You MUST always respond in English. Never respond in Chinese, Japanese, Korean, "
    "or any other non-English language unless the user explicitly asks you to."
)

# Build version
BUILD_VERSION = "007"

# In-memory session cache (primary, fast, message-to-message)
session_cache: Dict[str, List[dict]] = {}

# TTS rate limiting - ElevenLabs free tier allows 3 concurrent requests
tts_semaphore = asyncio.Semaphore(3)

async def call_elevenlabs_with_retry(client, voice_id, text, max_retries=3):
    """Call ElevenLabs API with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            async with tts_semaphore:  # Limit concurrent requests
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": ELEVENLABS_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_multilingual_v2"
                    }
                )
                return response
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                logger.warning(f"TTS attempt {attempt + 1} failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
    return None


def load_session(session_id: str) -> List[dict]:
    """Load conversation history. Memory first, GCS fallback."""
    if session_id in session_cache:
        return session_cache[session_id]
    try:
        blob = bucket.blob(f"sessions/{session_id}.json")
        if blob.exists():
            history = json.loads(blob.download_as_text())
            session_cache[session_id] = history
            return history
    except Exception as e:
        logger.error(f"Failed to load session from GCS {session_id}: {e}")
    return []


def save_session(session_id: str, messages: List[dict]):
    """Save conversation history to both memory and GCS."""
    session_cache[session_id] = messages
    try:
        blob = bucket.blob(f"sessions/{session_id}.json")
        blob.upload_from_string(json.dumps(messages), content_type="application/json")
    except Exception as e:
        logger.error(f"Failed to save session to GCS {session_id}: {e}")

# Auth storage - now persistent in GCS
TRUSTED_IPS_FILE = "trusted_ips.json"
TRUSTED_FINGERPRINTS_FILE = "trusted_fingerprints.json"

# Cookie secrets (in production these should be env vars)
COOKIE_SECRET = os.getenv("COOKIE_SECRET", "haley-cookie-secret-2026")

def get_trusted_ips():
    """Load trusted IPs from GCS"""
    try:
        blob = bucket.blob(TRUSTED_IPS_FILE)
        if blob.exists():
            data = blob.download_as_string()
            return set(json.loads(data))
        return set()
    except Exception as e:
        logger.error(f"Failed to load trusted IPs: {e}")
        return set()

def save_trusted_ips(ips):
    """Save trusted IPs to GCS"""
    try:
        blob = bucket.blob(TRUSTED_IPS_FILE)
        blob.upload_from_string(json.dumps(list(ips)), content_type="application/json")
    except Exception as e:
        logger.error(f"Failed to save trusted IPs: {e}")

def get_trusted_fingerprints():
    """Load trusted fingerprints from GCS"""
    try:
        blob = bucket.blob(TRUSTED_FINGERPRINTS_FILE)
        if blob.exists():
            data = blob.download_as_string()
            return set(json.loads(data))
        return set()
    except Exception as e:
        logger.error(f"Failed to load trusted fingerprints: {e}")
        return set()

def save_trusted_fingerprints(fingerprints):
    """Save trusted fingerprints to GCS"""
    try:
        blob = bucket.blob(TRUSTED_FINGERPRINTS_FILE)
        blob.upload_from_string(json.dumps(list(fingerprints)), content_type="application/json")
    except Exception as e:
        logger.error(f"Failed to save trusted fingerprints: {e}")

# Chat history storage
CHAT_HISTORY_FILE = "chat_history.json"

def get_chat_history():
    """Load chat history from GCS"""
    try:
        blob = bucket.blob(CHAT_HISTORY_FILE)
        if blob.exists():
            data = blob.download_as_string()
            return json.loads(data)
        return []
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
        return []

def save_chat_history(history):
    """Save chat history to GCS"""
    try:
        blob = bucket.blob(CHAT_HISTORY_FILE)
        blob.upload_from_string(json.dumps(history[-100:]), content_type="application/json")  # Keep last 100 messages
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")

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
    return {"status": "healthy", "version": BUILD_VERSION, "openclaw_url": OPENCLAW_URL}


@app.get("/api/history")
async def get_history(request: Request):
    """Get chat history"""
    history = get_chat_history()
    return {"messages": history}


@app.get("/api/check-auth")
async def check_auth(request: Request):
    """Check if user is trusted via IP, fingerprint, or cookie"""
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    fingerprint = request.headers.get("x-device-fingerprint", "")
    cookie_token = request.headers.get("x-auth-cookie", "")
    
    # Load from GCS
    trusted_ips = get_trusted_ips()
    trusted_fingerprints = get_trusted_fingerprints()
    
    # Check IP
    ip_trusted = client_ip in trusted_ips
    
    # Check fingerprint
    fp_trusted = fingerprint in trusted_fingerprints
    
    # Check cookie (simple validation)
    cookie_valid = False
    if cookie_token and cookie_token.startswith("haley-"):
        # In production, validate signature here
        cookie_valid = True
    
    trusted = ip_trusted or fp_trusted or cookie_valid
    
    logger.info(f"Auth check: IP={client_ip[:20]}... trusted={ip_trusted}, FP={fingerprint[:20]}... trusted={fp_trusted}, Cookie={cookie_valid}, Final={trusted}")
    
    return {
        "trusted": trusted,
        "ip": client_ip,
        "ip_trusted": ip_trusted,
        "fingerprint_trusted": fp_trusted,
        "cookie_valid": cookie_valid
    }


# Legacy endpoint for backward compatibility
@app.get("/api/check-ip")
async def check_ip(request: Request):
    """Check if IP is trusted (legacy endpoint)"""
    result = await check_auth(request)
    return {"trusted": result["trusted"], "ip": result["ip"]}


@app.post("/api/auth")
async def auth(request: Request):
    """Password gate authentication - remembers IP, fingerprint, and sets cookie"""
    data = await request.json()
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    fingerprint = data.get("fingerprint", "")
    
    if data.get("password") == AUTH_PASSWORD:
        # Load existing
        trusted_ips = get_trusted_ips()
        trusted_fingerprints = get_trusted_fingerprints()
        
        # Remember this IP
        trusted_ips.add(client_ip)
        save_trusted_ips(trusted_ips)
        
        # Remember fingerprint if provided
        if fingerprint:
            trusted_fingerprints.add(fingerprint)
            save_trusted_fingerprints(trusted_fingerprints)
            logger.info(f"Fingerprint added to trusted list")
        
        # Generate cookie token
        import hashlib
        import time
        cookie_token = f"haley-{hashlib.sha256(f'{client_ip}{fingerprint}{time.time()}'.encode()).hexdigest()[:32]}"
        
        logger.info(f"Auth success: IP {client_ip} added to trusted list")
        return {
            "token": "session-token",
            "valid": True,
            "ip": client_ip,
            "cookie": cookie_token,
            "fingerprint_saved": bool(fingerprint)
        }
    
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

        # Load conversation history (memory-first, GCS fallback)
        history = load_session(session_id)

        # Add user message to history
        history.append({"role": "user", "content": request.message})

        # Also save to global chat history for UI display
        global_history = get_chat_history()
        global_history.append({"role": "user", "content": request.message, "timestamp": str(uuid.uuid4())[:8]})

        # Build messages with system prompt + conversation history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

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
                    "messages": messages,
                    "stream": False,
                    "user": session_id
                }
            )

            if response.status_code == 200:
                data = response.json()
                assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Store assistant response in per-session history
                history.append({"role": "assistant", "content": assistant_message})

                # Keep history manageable (last 50 messages)
                if len(history) > 50:
                    history = history[-50:]

                # Save per-session history (memory + GCS)
                save_session(session_id, history)

                # Save to global chat history for UI display
                global_history.append({"role": "assistant", "content": assistant_message, "timestamp": str(uuid.uuid4())[:8]})
                save_chat_history(global_history)

                return {
                    "response": assistant_message,
                    "status": "success",
                    "session_id": session_id
                }
            else:
                logger.error(f"OpenClaw error: {response.status_code} - {response.text}")
                save_chat_history(history)  # Still save the user message
                return {
                    "response": f"OpenClaw returned error {response.status_code}. Please try again.",
                    "status": "error",
                    "session_id": session_id
                }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "response": "Something went wrong. Try again?",
            "status": "error",
            "session_id": request.session_id
        }


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
    """Text to speech using ElevenLabs with rate limiting"""
    try:
        if not ELEVENLABS_API_KEY:
            logger.error("ELEVENLABS_API_KEY not set")
            return JSONResponse(status_code=503, content={"error": "ElevenLabs API key not configured"})

        logger.info(f"TTS request: voice={request.voice_id}, text_len={len(request.text)}")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await call_elevenlabs_with_retry(client, request.voice_id, request.text)
            
            if response is None:
                logger.error("TTS failed after all retries")
                return JSONResponse(status_code=503, content={"error": "TTS service temporarily unavailable. Please try again in a few seconds."})

            logger.info(f"ElevenLabs response: status={response.status_code}")

            if response.status_code == 200:
                logger.info(f"TTS success: {len(response.content)} bytes")
                return StreamingResponse(
                    BytesIO(response.content),
                    media_type="audio/mpeg"
                )
            elif response.status_code == 429:
                logger.error(f"TTS rate limited: {response.text[:200]}")
                return JSONResponse(status_code=429, content={"error": "Too many requests. Please wait a few seconds and try again."})
            else:
                logger.error(f"TTS error: status={response.status_code}, body={response.text[:500]}")
                return JSONResponse(status_code=502, content={"error": f"TTS failed: {response.status_code}"})

    except Exception as e:
        logger.error(f"Speak error: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
