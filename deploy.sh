#!/bin/bash

# Haley Chat Deploy Script
# One command to deploy to Cloud Run

set -e

echo "🚀 Haley Chat Deploy Script"
echo ""

# Check if we're in the right directory
if [ ! -f "chat_service.py" ]; then
    echo "❌ Error: Run this from the navigator-service directory"
    exit 1
fi

# Get required env vars
echo "📋 Configuration:"
echo ""

read -p "Enter your OpenClaw URL (e.g., https://your-openclaw.tailnet.ts.net): " OPENCLAW_URL
read -p "Enter OpenAI API Key (for voice transcription): " OPENAI_API_KEY
read -p "Enter ElevenLabs API Key (for voice output): " ELEVENLABS_API_KEY

# Set defaults
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
if [ -z "$PROJECT_ID" ]; then
    read -p "Enter GCP Project ID: " PROJECT_ID
fi

echo ""
echo "🔧 Building and deploying..."
echo ""

# Submit build
gcloud builds submit \
    --config cloudbuild.yaml \
    --project "$PROJECT_ID" \
    --substitutions=_OPENCLAW_URL="$OPENCLAW_URL",_OPENAI_API_KEY="$OPENAI_API_KEY",_ELEVENLABS_API_KEY="$ELEVENLABS_API_KEY"

echo ""
echo "✅ Deployment complete!"
echo ""

# Get the URL
URL=$(gcloud run services describe haley-chat --region us-central1 --format 'value(status.url)' 2>/dev/null || echo "")

if [ -n "$URL" ]; then
    echo "🌐 Your chat is live at:"
    echo "   $URL"
    echo ""
    echo "📱 Bookmark this on your phone!"
fi