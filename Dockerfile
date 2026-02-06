FROM python:3.11-slim

# Install system dependencies and Node.js
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

# Install OpenClaw globally
RUN npm install -g openclaw

# Find where openclaw was installed and link it
RUN OPENCLAW_PATH=$(find /usr -name "openclaw" -type f 2>/dev/null | head -1) && \
    if [ -n "$OPENCLAW_PATH" ]; then \
        ln -s "$OPENCLAW_PATH" /usr/local/bin/openclaw; \
    fi && \
    ls -la /usr/local/bin/openclaw || echo "Not in /usr/local/bin" && \
    ls -la /usr/lib/node_modules/openclaw/bin/ 2>/dev/null || echo "Checking node_modules" && \
    find /usr -name "openclaw" -type f 2>/dev/null

# Add node_modules/.bin to PATH
ENV PATH="/usr/lib/node_modules/.bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy chat service code
COPY chat_service.py .
COPY static/ ./static/

# Copy OpenClaw config with OpenAI endpoint enabled
RUN mkdir -p /root/.openclaw
COPY openclaw.json /root/.openclaw/openclaw.json

# Copy startup script
COPY start-bundled.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8080

# Start both services
CMD ["/app/start.sh"]
