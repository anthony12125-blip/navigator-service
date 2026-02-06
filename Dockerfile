FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY chat_service.py .
COPY static/ static/

EXPOSE 8080

CMD ["python", "chat_service.py"]