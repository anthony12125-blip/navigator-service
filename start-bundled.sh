#!/bin/bash
set -e

# Start OpenClaw gateway in background
echo "Starting OpenClaw gateway..."
openclaw gateway &
OPENCLAW_PID=$!

# Wait for OpenClaw to be ready
echo "Waiting for OpenClaw to start..."
sleep 5

# Verify OpenClaw is running
if ! curl -s http://localhost:18789/health > /dev/null 2>&1; then
    echo "Warning: OpenClaw may not be ready yet, continuing anyway..."
fi

# Set environment variable for chat service
export OPENCLAW_URL=http://localhost:18789
export PORT=8080

echo "Starting chat service on port $PORT..."
python chat_service.py
