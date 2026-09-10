# Use official NVIDIA vLLM image - optimized for NVIDIA hardware
FROM nvcr.io/nvidia/vllm:25.11-py3

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the vLLM startup script
COPY launch_server.sh .

# Make startup script executable
RUN chmod +x launch_server.sh

# Create model directory
RUN mkdir -p /app/models

# Expose the service port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start vLLM's built-in OpenAI API server directly
CMD ["./launch_server.sh"]
