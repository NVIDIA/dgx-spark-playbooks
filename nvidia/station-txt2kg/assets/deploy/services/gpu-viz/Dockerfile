# Use latest NVIDIA PyG container which includes cuGraph and graph-related packages
FROM nvcr.io/nvidia/pyg:25.08-py3

# Ensure we're running as root for system package installation
USER root

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service code
COPY unified_gpu_service.py .
COPY pygraphistry_service.py .

# Create a non-root user for security (using a different UID to avoid conflicts)
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose unified service port
EXPOSE 8080

# Health check for unified service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start unified service
CMD ["python", "unified_gpu_service.py"] 