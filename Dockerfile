FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY chat_service.py .
COPY static/ static/

# Expose port
EXPOSE 8080

# Run
CMD ["python", "chat_service.py"]