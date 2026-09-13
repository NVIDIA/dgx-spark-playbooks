import { NextRequest, NextResponse } from 'next/server';
import { PineconeService } from '@/lib/pinecone';

/**
 * Get Pinecone vector database stats
 */
export async function GET() {
  try {
    // Initialize Pinecone service
    const pineconeService = PineconeService.getInstance();
    
    // We can now directly call getStats() which handles initialization and error recovery
    const stats = await pineconeService.getStats();
    
    return NextResponse.json({
      ...stats,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error getting Pinecone stats:', error);
    
    // Return a successful response with error information
    // This prevents the UI from breaking when Pinecone is unavailable
    let errorMessage = error instanceof Error ? error.message : String(error);
    
    // More specific error message for 404 errors
    if (errorMessage.includes('404')) {
      errorMessage = 'Pinecone server returned 404. The server may not be running or the index does not exist.';
    }
    
    return NextResponse.json(
      { 
        error: `Failed to get Pinecone stats: ${errorMessage}`,
        totalVectorCount: 0,
        source: 'error',
        httpHealthy: false,
        timestamp: new Date().toISOString()
      },
      { status: 200 } // Use 200 instead of 500 to avoid UI errors
    );
  }
} 