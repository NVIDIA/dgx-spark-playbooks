import { NextRequest, NextResponse } from 'next/server';
import { PineconeService } from '@/lib/pinecone';

/**
 * Clear all data from the Pinecone vector database
 * POST /api/pinecone-diag/clear
 */
export async function POST() {
  // Get the Pinecone service instance
  const pineconeService = PineconeService.getInstance();
  
  // Clear all vectors from the database
  const deleteSuccess = await pineconeService.deleteAllEntities();
  
  // Get updated stats after clearing
  const stats = await pineconeService.getStats();
  
  // Return response based on operation success
  return NextResponse.json({
    success: deleteSuccess,
    message: deleteSuccess 
      ? 'Successfully cleared all data from Pinecone vector database'
      : 'Failed to clear Pinecone database - service may not be available',
    totalVectorCount: stats.totalVectorCount || 0,
    httpHealthy: stats.httpHealthy || false
  });
} 