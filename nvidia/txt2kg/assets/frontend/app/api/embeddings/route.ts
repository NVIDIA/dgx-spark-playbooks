import { NextRequest, NextResponse } from 'next/server';
import { EmbeddingsService } from '@/lib/embeddings';
import { QdrantService } from '@/lib/qdrant';

/**
 * Generate embeddings for text chunks and store them in Qdrant
 */
export async function POST(request: NextRequest) {
  try {
    const { documentId, content, documentName } = await request.json();
    
    if (!content) {
      return NextResponse.json(
        { error: 'Document content is required' }, 
        { status: 400 }
      );
    }
    
    // Initialize embedding service
    const embeddingsService = EmbeddingsService.getInstance();
    
    // Log which provider we're using
    console.log(`Using embeddings provider: ${process.env.EMBEDDINGS_PROVIDER || 'local'}`);
    
    // Generate chunks from content
    const chunkSize = 200; // Size of each text chunk
    const chunks = generateChunks(content, chunkSize);
    console.log(`Generated ${chunks.length} chunks from document`);
    
    // Create unique IDs for each chunk based on document name and chunk index
    const docPrefix = documentName ? 
      documentName.replace(/[^a-zA-Z0-9]/g, '_').substring(0, 20) : 
      documentId ? documentId : 'doc';
    
    const chunkIds = chunks.map((_, index) => `${docPrefix}_chunk_${index}`);
    
    // Generate embeddings for chunks
    console.log('Generating embeddings for chunks...');
    const embeddings = await embeddingsService.encode(chunks);
    console.log(`Generated ${embeddings.length} embeddings`);

    // Initialize QdrantService
    const pineconeService = QdrantService.getInstance();

    // Check if Qdrant server is running
    const isPineconeRunning = await pineconeService.isQdrantRunning();
    if (!isPineconeRunning) {
      return NextResponse.json(
        { error: 'Qdrant server is not available. Please make sure it is running.' },
        { status: 503 }
      );
    }
    
    if (!pineconeService.isInitialized()) {
      try {
        await pineconeService.initialize();
      } catch (initError) {
        console.error('Error initializing Pinecone:', initError);
        return NextResponse.json(
          { error: `Failed to initialize Pinecone: ${initError instanceof Error ? initError.message : String(initError)}` },
          { status: 500 }
        );
      }
    }
    
    // Create maps for embeddings and text content
    const entityEmbeddings = new Map<string, number[]>();
    const textContent = new Map<string, string>();
    
    // Populate the maps
    for (let i = 0; i < chunkIds.length; i++) {
      entityEmbeddings.set(chunkIds[i], embeddings[i]);
      textContent.set(chunkIds[i], chunks[i]);
    }
    
    // Store embeddings in PineconeService with retry logic
    try {
      await pineconeService.storeEmbeddings(entityEmbeddings, textContent);
    } catch (storeError) {
      console.error('Error storing embeddings in Pinecone:', storeError);
      return NextResponse.json(
        { error: `Failed to store embeddings in Pinecone: ${storeError instanceof Error ? storeError.message : String(storeError)}` },
        { status: 500 }
      );
    }
    
    return NextResponse.json({
      success: true,
      documentId: documentId || 'unnamed',
      chunks: chunks.length,
      embeddings: embeddings.length
    });
    
  } catch (error) {
    console.error('Error generating embeddings:', error);
    return NextResponse.json(
      { error: `Failed to generate embeddings: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}

/**
 * Generate chunks from text content
 * @param content Text content
 * @param chunkSize Size of each chunk
 * @param overlap Overlap between chunks
 * @returns Array of text chunks
 */
function generateChunks(content: string, chunkSize: number, overlap: number = 50): string[] {
  const chunks: string[] = [];
  const sentences = content.split(/(?<=[.!?])\s+/);
  
  let currentChunk = '';
  for (const sentence of sentences) {
    // If adding this sentence would make the chunk too long, save the current chunk and start a new one
    if (currentChunk.length + sentence.length > chunkSize && currentChunk.length > 0) {
      chunks.push(currentChunk.trim());
      // Take the last part of the current chunk as overlap for the next chunk
      const words = currentChunk.split(' ');
      currentChunk = words.slice(Math.max(0, words.length - overlap)).join(' ');
    }
    
    currentChunk += ' ' + sentence;
  }
  
  // Add the last chunk if it's not empty
  if (currentChunk.trim().length > 0) {
    chunks.push(currentChunk.trim());
  }
  
  return chunks;
} 