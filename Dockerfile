FROM python:3.11-slim

# Install Node.js 20.x properly
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Verify Node.js
RUN node --version && npm --version

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy chat service code
COPY chat_service.py .
COPY static/ ./static/

# Copy OpenClaw config (for local reference only - OpenClaw runs separately)
RUN mkdir -p /root/.openclaw
COPY openclaw.json /root/.openclaw/openclaw.json

# Expose port
EXPOSE 8080

# Run chat service only (OpenClaw runs locally on Mac)
CMD ["python", "chat_service.py"]
