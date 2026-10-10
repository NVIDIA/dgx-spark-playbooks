# Stage 1: Build React UI
FROM node:20-slim AS ui-build
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json* ./
RUN npm install
COPY ui/ ./
RUN npm run build

# Stage 2: Python backend + static UI
FROM python:3.12-slim
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
COPY pyproject.toml ./
RUN uv pip install --system -e .

# Copy backend source
COPY src/ ./src/
COPY config.yaml ./

# Copy built UI
COPY --from=ui-build /app/ui/dist ./ui/dist

EXPOSE 8234

CMD ["python", "-m", "src.server.app"]
