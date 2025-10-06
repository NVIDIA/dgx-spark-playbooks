/**
 * Client-side initialization utilities
 * This file contains functions for initializing the application on the client side
 */

/**
 * Initialize default database settings if not already set
 * Called before syncing with server to ensure defaults are available
 */
export function initializeDefaultSettings() {
  if (typeof window === 'undefined') {
    return; // Only run on client side
  }

  // Set default graph DB type to ArangoDB if not set
  if (!localStorage.getItem('graph_db_type')) {
    localStorage.setItem('graph_db_type', 'arangodb');
  }

  // Set default ArangoDB settings if not set
  if (!localStorage.getItem('arango_url')) {
    localStorage.setItem('arango_url', 'http://localhost:8529');
  }

  if (!localStorage.getItem('arango_db')) {
    localStorage.setItem('arango_db', 'txt2kg');
  }
}

/**
 * Synchronize settings from localStorage with the server
 * Called on app initialization to ensure server has access to client settings
 */
export async function syncSettingsWithServer() {
  if (typeof window === 'undefined') {
    return; // Only run on client side
  }

  // Initialize default settings first
  initializeDefaultSettings();

  // Collect all relevant settings from localStorage
  const settings: Record<string, string> = {};
  
  // NVIDIA API settings
  const nvidiaEmbeddingsModel = localStorage.getItem('nvidia_embeddings_model');
  if (nvidiaEmbeddingsModel) {
    settings.nvidia_embeddings_model = nvidiaEmbeddingsModel;
  }
  
  const embeddingsProvider = localStorage.getItem('embeddings_provider');
  if (embeddingsProvider) {
    settings.embeddings_provider = embeddingsProvider;
  }
  
  // Graph Database selection
  const graphDbType = localStorage.getItem('graph_db_type');
  if (graphDbType) {
    settings.graph_db_type = graphDbType;
  }
  
  // Neo4j settings
  const neo4jUrl = localStorage.getItem('neo4j_url');
  if (neo4jUrl) {
    settings.neo4j_url = neo4jUrl;
  }
  
  const neo4jUser = localStorage.getItem('neo4j_user');
  if (neo4jUser) {
    settings.neo4j_user = neo4jUser;
  }
  
  const neo4jPassword = localStorage.getItem('neo4j_password');
  if (neo4jPassword) {
    settings.neo4j_password = neo4jPassword;
  }
  
  // ArangoDB settings
  const arangoUrl = localStorage.getItem('arango_url');
  if (arangoUrl) {
    settings.arango_url = arangoUrl;
  }
  
  const arangoDb = localStorage.getItem('arango_db');
  if (arangoDb) {
    settings.arango_db = arangoDb;
  }
  
  const arangoUser = localStorage.getItem('arango_user');
  if (arangoUser) {
    settings.arango_user = arangoUser;
  }
  
  const arangoPassword = localStorage.getItem('arango_password');
  if (arangoPassword) {
    settings.arango_password = arangoPassword;
  }
  
  // xAI API settings
  const xaiApiKey = localStorage.getItem('XAI_API_KEY');
  if (xaiApiKey) {
    settings.XAI_API_KEY = xaiApiKey;
  }
  
  // NVIDIA Nemotron API key
  const nvidiaApiKey = localStorage.getItem('NVIDIA_API_KEY');
  if (nvidiaApiKey) {
    settings.NVIDIA_API_KEY = nvidiaApiKey;
  }
  
  // Pinecone settings
  const pineconeApiKey = localStorage.getItem('pinecone_api_key');
  if (pineconeApiKey) {
    settings.pinecone_api_key = pineconeApiKey;
  }
  
  const pineconeEnvironment = localStorage.getItem('pinecone_environment');
  if (pineconeEnvironment) {
    settings.pinecone_environment = pineconeEnvironment;
  }
  
  const pineconeIndex = localStorage.getItem('pinecone_index');
  if (pineconeIndex) {
    settings.pinecone_index = pineconeIndex;
  }
  
  // Skip the API call if there are no settings to sync
  if (Object.keys(settings).length === 0) {
    return;
  }
  
  // Send settings to server
  try {
    const response = await fetch('/api/settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ settings }),
    });
    
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
    }
    
    console.log('Client settings synchronized with server');
  } catch (error) {
    console.error('Failed to sync settings with server:', error);
  }
} 