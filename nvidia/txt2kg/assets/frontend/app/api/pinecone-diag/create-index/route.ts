import { NextResponse } from 'next/server';
import { PineconeService } from '@/lib/pinecone';

/**
 * Create Pinecone index API endpoint
 * POST /api/pinecone-diag/create-index
 */
export async function POST() {
  try {
    // Get the Pinecone service instance
    const pineconeService = PineconeService.getInstance();
    
    // Force re-initialization to create the index
    (pineconeService as any).initialized = false;
    await pineconeService.initialize();
    
    // Check if initialization was successful by getting stats
    const stats = await pineconeService.getStats();
    
    return NextResponse.json({
      success: true,
      message: 'Pinecone index created successfully',
      httpHealthy: stats.httpHealthy || false
    });
  } catch (error) {
    console.error('Error creating Pinecone index:', error);
    
    return NextResponse.json(
      { 
        success: false,
        error: `Failed to create Pinecone index: ${error instanceof Error ? error.message : String(error)}`
      },
      { status: 500 }
    );
  }
} 