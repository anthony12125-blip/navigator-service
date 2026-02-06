#!/bin/bash
set -e

# Find openclaw binary
OPENCLAW_BIN=$(which openclaw 2>/dev/null || find /usr -name "openclaw" -type f 2>/dev/null | head -1 || echo "")

if [ -z "$OPENCLAW_BIN" ]; then
    echo "Error: openclaw binary not found"
    # Try to find node and run openclaw directly from node_modules
    if [ -f /usr/lib/node_modules/openclaw/dist/index.js ]; then
        echo "Attempting to run from node_modules..."
        node /usr/lib/node_modules/openclaw/dist/index.js gateway &
        OPENCLAW_PID=$!
    else
        echo "Falling back to chat service only mode..."
        export PORT=8080
        python chat_service.py
        exit 0
    fi
else
    echo "Found openclaw at: $OPENCLAW_BIN"
    # Start OpenClaw gateway in background
    echo "Starting OpenClaw gateway..."
    "$OPENCLAW_BIN" gateway &
    OPENCLAW_PID=$!
fi

# Wait for OpenClaw to be ready
echo "Waiting for OpenClaw to start..."
sleep 5

# Verify OpenClaw is running
if curl -s http://localhost:18789/health > /dev/null 2>&1; then
    echo "OpenClaw is running!"
else
    echo "Warning: OpenClaw may not be ready yet, continuing anyway..."
fi

# Set environment variable for chat service
export OPENCLAW_URL=http://localhost:18789
export PORT=8080

echo "Starting chat service on port $PORT..."
python chat_service.py
