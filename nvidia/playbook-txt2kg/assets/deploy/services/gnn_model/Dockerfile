FROM nvcr.io/nvidia/pyg:26.09-py3

WORKDIR /app

# RelBench is unused by this service and can constrain shared ML packages.
COPY requirements.txt .
RUN pip uninstall -y relbench && \
    pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p /app/models

# Copy application code
COPY app.py /app/

# Set environment variables
ENV MODEL_PATH=/app/models/tech-qa-model.pt
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Expose the port
EXPOSE 5000

# Run the service with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
