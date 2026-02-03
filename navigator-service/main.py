"""
Navigator Service - Standalone WebRTC Browser Streaming
Separated from main HaleyOS backend for stability
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# VM Orchestrator
# ============================================================================

class VMOrchestrator:
    """Manages VM pool and session allocation"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.project_id = os.getenv("GCP_PROJECT_ID", "gen-lang-client-0013187443")
        self.zone = os.getenv("GCP_ZONE", "us-central1-a")
        self.pool_prefix = "haleyos-pool"
        
    async def create_session(self, user_id: str, template: str = "standard") -> Dict[str, Any]:
        """Create a new VM session"""
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        token = uuid.uuid4().hex
        
        # Try to find an available VM from pool
        vm_info = await self._allocate_vm(template)
        
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "token": token,
            "template": template,
            "status": "ready" if vm_info else "creating",
            "vm_name": vm_info.get("name") if vm_info else None,
            "external_ip": vm_info.get("external_ip") if vm_info else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id}")
        return session
    
    async def _allocate_vm(self, template: str) -> Optional[Dict[str, Any]]:
        """Allocate a VM from the pool"""
        try:
            from google.cloud import compute_v1
            
            client = compute_v1.InstancesClient()
            request = compute_v1.ListInstancesRequest(
                project=self.project_id,
                zone=self.zone,
                filter=f"name:{self.pool_prefix}-* AND status=RUNNING"
            )
            
            instances = list(client.list(request=request))
            
            # Find first available VM matching template
            template_short = template[:8].lower().replace("-", "").replace("_", "")
            for instance in instances:
                if template_short in instance.name.lower():
                    external_ip = None
                    for interface in instance.network_interfaces:
                        for config in interface.access_configs:
                            if config.nat_i_p:
                                external_ip = config.nat_i_p
                                break
                    
                    if external_ip:
                        return {
                            "name": instance.name,
                            "external_ip": external_ip
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to allocate VM: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def validate_token(self, session_id: str, token: str) -> bool:
        """Validate session token"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        return session.get("token") == token
    
    async def update_activity(self, session_id: str):
        """Update last activity timestamp"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = datetime.now(timezone.utc).isoformat()
    
    async def terminate_session(self, session_id: str):
        """Terminate a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Terminated session {session_id}")


# Global orchestrator
orchestrator = VMOrchestrator()


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Navigator Service starting...")
    yield
    logger.info("Navigator Service shutting down...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Navigator Service",
    description="WebRTC Browser Streaming for HaleyOS",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://haley-front-end.web.app",
        "https://haley-front-end.firebaseapp.com",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Models
# ============================================================================

class CreateSessionRequest(BaseModel):
    template: str = "standard"
    user_id: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    status: str
    token: Optional[str] = None
    external_ip: Optional[str] = None
    vm_name: Optional[str] = None


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "navigator"}


@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new Navigator session"""
    user_id = request.user_id or f"anon_{uuid.uuid4().hex[:8]}"
    session = await orchestrator.create_session(user_id, request.template)
    return SessionResponse(
        session_id=session["session_id"],
        status=session["status"],
        token=session["token"],
        external_ip=session.get("external_ip"),
        vm_name=session.get("vm_name")
    )


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session status"""
    session = orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=session["session_id"],
        status=session["status"],
        external_ip=session.get("external_ip"),
        vm_name=session.get("vm_name")
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Terminate a session"""
    session = orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await orchestrator.terminate_session(session_id)
    return {"status": "terminated"}


# ============================================================================
# WebSocket Streaming
# ============================================================================

@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...)
):
    """WebSocket endpoint for video streaming"""
    
    # Validate session
    if not orchestrator.validate_token(session_id, token):
        await websocket.close(code=4001, reason="Invalid session or token")
        return
    
    session = orchestrator.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    await websocket.accept()
    logger.info(f"WebSocket connected for session {session_id}")
    
    try:
        vm_ip = session.get("external_ip")
        
        if vm_ip:
            # Try WebRTC via Janus
            success = await _run_janus_session(websocket, session_id, vm_ip)
            
            if not success:
                # Fall back to canvas frames
                logger.warning(f"Janus failed for {session_id}, using canvas fallback")
                await _run_canvas_session(websocket, session_id)
        else:
            # No VM, use canvas mode
            await _run_canvas_session(websocket, session_id)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        await orchestrator.update_activity(session_id)


async def _run_janus_session(websocket: WebSocket, session_id: str, vm_ip: str) -> bool:
    """
    Run a full Janus WebRTC session.
    Returns True if successful, False if should fall back to canvas.
    """
    try:
        import aiohttp
        janus_url = f"ws://{vm_ip}:8188"
        
        async with aiohttp.ClientSession() as http_session:
            async with http_session.ws_connect(
                janus_url, 
                timeout=aiohttp.ClientTimeout(total=10),
                protocols=['janus-protocol']
            ) as janus_ws:
                
                # Create Janus session
                await janus_ws.send_json({
                    "janus": "create",
                    "transaction": f"create_{session_id}"
                })
                create_resp = await janus_ws.receive_json()
                if create_resp.get("janus") != "success":
                    logger.error(f"Janus create failed: {create_resp}")
                    return False
                janus_session_id = create_resp["data"]["id"]
                
                # Attach to streaming plugin
                await janus_ws.send_json({
                    "janus": "attach",
                    "session_id": janus_session_id,
                    "plugin": "janus.plugin.streaming",
                    "transaction": f"attach_{session_id}"
                })
                attach_resp = await janus_ws.receive_json()
                if attach_resp.get("janus") != "success":
                    logger.error(f"Janus attach failed: {attach_resp}")
                    return False
                handle_id = attach_resp["data"]["id"]
                
                # Send watch request
                await janus_ws.send_json({
                    "janus": "message",
                    "session_id": janus_session_id,
                    "handle_id": handle_id,
                    "body": {"request": "watch", "id": 1},
                    "transaction": f"watch_{session_id}"
                })
                
                # Get Janus offer
                janus_offer = None
                for _ in range(5):
                    resp = await janus_ws.receive_json()
                    if "jsep" in resp:
                        janus_offer = resp["jsep"]
                        break
                    if resp.get("janus") == "error":
                        logger.error(f"Janus error: {resp}")
                        return False
                
                if not janus_offer:
                    logger.error("No offer from Janus")
                    return False
                
                # Send offer to browser
                await websocket.send_json({
                    "type": "offer",
                    "session_id": session_id,
                    "payload": {"sdp": janus_offer}
                })
                
                # Wait for browser answer
                answer_data = await websocket.receive_json()
                if answer_data.get("type") != "answer":
                    logger.error(f"Expected answer, got {answer_data.get('type')}")
                    return False
                
                browser_answer = answer_data.get("payload", {}).get("sdp")
                
                # Send answer to Janus
                await janus_ws.send_json({
                    "janus": "message",
                    "session_id": janus_session_id,
                    "handle_id": handle_id,
                    "body": {"request": "start"},
                    "jsep": browser_answer,
                    "transaction": f"start_{session_id}"
                })
                
                await janus_ws.receive_json()  # ack
                logger.info(f"Janus streaming started for {session_id}")
                
                # Keep both connections alive
                # Handle messages from browser and keepalive to Janus
                keepalive_task = asyncio.create_task(
                    _janus_keepalive(janus_ws, janus_session_id)
                )
                
                try:
                    while True:
                        # Wait for messages from browser
                        data = await websocket.receive_json()
                        msg_type = data.get("type")
                        
                        if msg_type == "ping":
                            await websocket.send_json({"type": "pong"})
                            await orchestrator.update_activity(session_id)
                        
                        elif msg_type == "ice_candidate":
                            # Forward ICE candidate to Janus
                            candidate = data.get("payload", {}).get("candidate")
                            if candidate:
                                await janus_ws.send_json({
                                    "janus": "trickle",
                                    "session_id": janus_session_id,
                                    "handle_id": handle_id,
                                    "candidate": candidate,
                                    "transaction": f"trickle_{uuid.uuid4().hex[:8]}"
                                })
                        
                        elif msg_type == "input":
                            # Handle input events (mouse, keyboard)
                            logger.debug(f"Input event for {session_id}")
                            
                finally:
                    keepalive_task.cancel()
                    
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Janus session error: {e}")
        return False
    
    return True


async def _janus_keepalive(janus_ws, janus_session_id: int):
    """Send keepalive to Janus every 25 seconds"""
    try:
        while True:
            await asyncio.sleep(25)
            await janus_ws.send_json({
                "janus": "keepalive",
                "session_id": janus_session_id,
                "transaction": f"keepalive_{uuid.uuid4().hex[:8]}"
            })
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Keepalive error: {e}")


async def _run_canvas_session(websocket: WebSocket, session_id: str):
    """Run canvas frame fallback mode"""
    await websocket.send_json({
        "type": "fallback",
        "session_id": session_id,
        "payload": {"mode": "canvas_frames"}
    })
    
    # Start sending mock frames
    frame_task = asyncio.create_task(_send_mock_frames(websocket, session_id))
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                await orchestrator.update_activity(session_id)
    except:
        pass
    finally:
        frame_task.cancel()


async def _send_mock_frames(websocket: WebSocket, session_id: str):
    """Send mock browser frames"""
    import base64
    from io import BytesIO
    
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("PIL not available for mock frames")
        return
    
    frame_count = 0
    
    try:
        while True:
            # Create mock browser image
            img = Image.new('RGB', (1920, 1080), color=(30, 30, 40))
            draw = ImageDraw.Draw(img)
            
            # Browser chrome
            draw.rectangle([0, 0, 1920, 80], fill=(50, 50, 60))
            draw.ellipse([20, 25, 50, 55], fill=(255, 95, 86))
            draw.ellipse([60, 25, 90, 55], fill=(255, 189, 46))
            draw.ellipse([100, 25, 130, 55], fill=(39, 201, 63))
            draw.rectangle([200, 20, 1720, 60], fill=(40, 40, 50))
            
            # URL bar text
            try:
                font = ImageFont.load_default()
            except:
                font = None
            draw.text((220, 30), "https://haleyos.local", fill=(180, 180, 180), font=font)
            
            # Content area
            timestamp = datetime.now().strftime("%H:%M:%S")
            draw.text((860, 500), f"HaleyOS Browser", fill=(200, 200, 200), font=font)
            draw.text((840, 550), f"VM Stream Active | {timestamp}", fill=(150, 150, 150), font=font)
            draw.text((850, 600), f"Frame {frame_count}", fill=(100, 100, 100), font=font)
            
            # Encode as JPEG
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=70)
            frame_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Send frame
            await websocket.send_json({
                "type": "frame",
                "session_id": session_id,
                "payload": {
                    "data": frame_data,
                    "width": 1920,
                    "height": 1080,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
            
            frame_count += 1
            await asyncio.sleep(0.5)  # 2 FPS
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Mock frame error: {e}")


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
