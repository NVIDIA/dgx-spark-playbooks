#!/bin/sh

# Script to initialize Qdrant collection at container startup
echo "Initializing Qdrant collection..."

# Wait for the Qdrant service to become available
echo "Waiting for Qdrant service to start..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
  if curl -s http://qdrant:6333/healthz > /dev/null; then
    echo "Qdrant service is up!"
    break
  fi
  echo "Waiting for Qdrant service (attempt $attempt/$max_attempts)..."
  attempt=$((attempt + 1))
  sleep 2
done

if [ $attempt -gt $max_attempts ]; then
  echo "Timed out waiting for Qdrant service"
  exit 1
fi

# Check if collection already exists
echo "Checking if collection 'entity-embeddings' exists..."
COLLECTION_EXISTS=$(curl -s http://qdrant:6333/collections/entity-embeddings | grep -c '"status":"ok"' || echo "0")

if [ "$COLLECTION_EXISTS" -gt "0" ]; then
  echo "Collection 'entity-embeddings' already exists, skipping creation"
else
  # Create the collection
  echo "Creating collection 'entity-embeddings'..."
  curl -X PUT "http://qdrant:6333/collections/entity-embeddings" \
    -H "Content-Type: application/json" \
    -d '{
      "vectors": {
        "size": 384,
        "distance": "Cosine"
      }
    }'

  if [ $? -eq 0 ]; then
    echo "✅ Collection 'entity-embeddings' created successfully"
  else
    echo "❌ Failed to create collection"
    exit 1
  fi
fi

echo "Qdrant initialization complete"
