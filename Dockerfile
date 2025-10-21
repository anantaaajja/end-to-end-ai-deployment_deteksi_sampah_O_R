# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode
COPY . .

# Port default Hugging Face Spaces
EXPOSE 7860

# Jalankan FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]