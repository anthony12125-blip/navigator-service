# Navigator Service

Standalone WebRTC browser streaming service for HaleyOS.

## Overview

Navigator provides real-time browser control via WebRTC streaming. It runs as a separate service from the main HaleyOS backend for stability - if WhatsApp Baileys or other services crash, Navigator keeps running.

## Architecture

```
Browser <--WebSocket--> Navigator Service <--WebSocket--> Janus on VM
                              |
                              v
                        GCP VM Pool
```

## Endpoints

- `GET /health` - Health check
- `POST /sessions` - Create new session
- `GET /sessions/{id}` - Get session status
- `DELETE /sessions/{id}` - Terminate session
- `WS /ws/stream/{id}?token=xxx` - WebSocket for video stream

## Environment Variables

- `GCP_PROJECT_ID` - Google Cloud project ID
- `GCP_ZONE` - Zone for VMs (default: us-central1-a)

## Local Development

```bash
pip install -r requirements.txt
python main.py
```

## Deployment

Push to main branch triggers Cloud Build:

```bash
git push origin main
```

Or manual deploy:

```bash
gcloud builds submit --config cloudbuild.yaml
```

## Frontend Integration

Update frontend to use Navigator service URL:

```typescript
const NAVIGATOR_URL = process.env.NEXT_PUBLIC_NAVIGATOR_URL || 
  'https://navigator-service-xxxxx.run.app';
```

## Key Differences from Main Backend

1. **No Baileys** - No WhatsApp, no crashes from that
2. **Long-lived WebSockets** - Janus connection stays open for ICE
3. **Dedicated scaling** - Scales independently based on Navigator load
4. **Simpler** - Only does one thing: browser streaming
