FROM nvcr.io/nvidia/pyg:25.03-py3

WORKDIR /app

# Install Flask and other required packages
RUN pip install --no-cache-dir \
    flask==2.0.1 \
    gunicorn==23.0.0 \
    tqdm

# Create model directory
RUN mkdir -p /app/models

# Copy application code
COPY services/gnn_model/app.py /app/

# Set environment variables
ENV MODEL_PATH=/app/models/tech-qa-model.pt
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Expose the port
EXPOSE 5000

# Run the service with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 