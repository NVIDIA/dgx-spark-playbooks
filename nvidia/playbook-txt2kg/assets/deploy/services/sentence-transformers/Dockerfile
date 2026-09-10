FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py /app/

# Set default model name
ENV MODEL_NAME="all-MiniLM-L6-v2"
ENV TRANSFORMERS_CACHE="/app/.cache"

# Pre-download the model during build for faster startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# Expose the port
EXPOSE 80

# Use Gunicorn for better performance
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "1", "--threads", "8", "app:app"] 