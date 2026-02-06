FROM node:20-slim AS node-base

# Stage 2: Python + Node combined
FROM python:3.11-slim

# Copy Node.js from node image
COPY --from=node-base /usr/local/bin/node /usr/local/bin/
COPY --from=node-base /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node-base /usr/local/bin/npm /usr/local/bin/
COPY --from=node-base /usr/local/bin/npx /usr/local/bin/

# Create node symlinks
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm-cli.js 2>/dev/null || true

# Install OpenClaw
RUN npm install -g openclaw

# Find and link openclaw
RUN find /usr -name "openclaw" -type f 2>/dev/null && \
    ls -la /usr/local/lib/node_modules/openclaw/ 2>/dev/null || true

# Add to PATH
ENV PATH="/usr/local/bin:/usr/local/lib/node_modules/.bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy chat service code
COPY chat_service.py .
COPY static/ ./static/

# Copy OpenClaw config
RUN mkdir -p /root/.openclaw
COPY openclaw.json /root/.openclaw/openclaw.json

# Copy startup script
COPY start-bundled.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8080

CMD ["/app/start.sh"]
