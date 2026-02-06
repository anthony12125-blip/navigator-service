"""
Haley Chat Service - Simplified chat with Twilio voice
Deploy to Cloud Run for a single URL endpoint
"""

import os
import asyncio
import logging
from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import httpx
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config from environment
OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://localhost:18789")
OPENCLAW_TOKEN = os.getenv("OPENCLAW_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
USER_PHONE_NUMBER = os.getenv("USER_PHONE_NUMBER", "+14075162030")

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


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send message to OpenClaw and get response"""
    try:
        logger.info(f"Chat message: {request.message[:50]}...")
        
        # Simple HTTP POST to OpenClaw instead of WebSocket
        # Using the chat completions endpoint if available
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OPENCLAW_URL}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENCLAW_TOKEN}" if OPENCLAW_TOKEN else ""
                },
                json={
                    "model": "haley",
                    "messages": [{"role": "user", "content": request.message}],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"response": assistant_message, "status": "success"}
            else:
                # Fallback - try WebSocket approach or return error
                logger.error(f"OpenClaw error: {response.status_code}")
                return {"response": "I'm having trouble connecting. Try again?", "status": "error"}
                
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": "Something went wrong. Try again?", "status": "error"}


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


@app.post("/api/call")
async def make_call():
    """Initiate a Twilio voice call"""
    try:
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
            return {"error": "Twilio not configured"}
        
        from twilio.rest import Client
        
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Create a call
        call = client.calls.create(
            to=USER_PHONE_NUMBER,
            from_=TWILIO_FROM_NUMBER,
            url=f"{os.getenv('PUBLIC_URL', '')}/twilio/voice"
        )
        
        logger.info(f"Call initiated: {call.sid}")
        return {"success": True, "call_sid": call.sid}
        
    except Exception as e:
        logger.error(f"Call error: {e}")
        return {"error": str(e)}


@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """Twilio webhook for voice calls"""
    from twilio.twiml.voice_response import VoiceResponse, Connect
    
    response = VoiceResponse()
    
    # Connect to a Stream for real-time audio
    connect = Connect()
    connect.stream(url=f"wss://{request.url.hostname}/twilio/stream")
    response.append(connect)
    
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/twilio/stream")
async def twilio_stream(websocket):
    """WebSocket for real-time Twilio audio streaming"""
    import websockets
    
    await websocket.accept()
    
    try:
        # Connect to OpenClaw
        openclaw_ws = await websockets.connect(
            f"{OPENCLAW_URL.replace('http', 'ws')}/ws"
        )
        
        async def forward_to_openclaw():
            """Forward audio from Twilio to OpenClaw"""
            async for message in websocket.iter_text():
                data = json.loads(message)
                if data.get("event") == "media":
                    # Audio chunk from Twilio
                    audio_data = base64.b64decode(data["media"]["payload"])
                    # TODO: Stream to OpenClaw or buffer for transcription
                    
        async def forward_to_twilio():
            """Forward responses from OpenClaw to Twilio"""
            async for message in openclaw_ws:
                data = json.loads(message)
                if data.get("type") == "token":
                    # Send text response back to Twilio for TTS
                    pass
                    
        # Run both directions
        await asyncio.gather(
            forward_to_openclaw(),
            forward_to_twilio()
        )
        
    except Exception as e:
        logger.error(f"Twilio stream error: {e}")
    finally:
        await websocket.close()


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)