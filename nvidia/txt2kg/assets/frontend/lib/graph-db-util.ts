//
// SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import { GraphDBService, GraphDBType } from './graph-db-service';
import { Neo4jService } from './neo4j';
import { ArangoDBService } from './arangodb';

/**
 * Get the default graph database type from environment or fallback to arangodb
 * Note: This is called at runtime, not build time, so process.env should be available
 */
function getDefaultGraphDbType(): GraphDBType {
  const envType = process.env.GRAPH_DB_TYPE;
  console.log(`[graph-db-util] getDefaultGraphDbType: env=${envType}`);
  return (envType as GraphDBType) || 'arangodb';
}

/**
 * Get the appropriate graph database service based on the graph database type.
 * This is useful for API routes that need direct access to a specific graph database.
 * 
 * @param graphDbType - The type of graph database to use (defaults to GRAPH_DB_TYPE env var)
 */
export function getGraphDbService(graphDbType?: GraphDBType) {
  const dbType = graphDbType || getDefaultGraphDbType();
  
  if (dbType === 'neo4j') {
    return Neo4jService.getInstance();
  } else if (dbType === 'arangodb') {
    return ArangoDBService.getInstance();
  } else {
    // Default based on environment
    return getDefaultGraphDbType() === 'neo4j' 
      ? Neo4jService.getInstance() 
      : ArangoDBService.getInstance();
  }
}

/**
 * Initialize the graph database directly (not using GraphDBService).
 * This is useful for API routes that need direct access to a specific graph database.
 * 
 * @param graphDbType - The type of graph database to use (defaults to GRAPH_DB_TYPE env var)
 */
export async function initializeGraphDb(graphDbType?: GraphDBType): Promise<void> {
  const dbType = graphDbType || getDefaultGraphDbType();
  const service = getGraphDbService(dbType);
  
  if (dbType === 'neo4j') {
    // Get Neo4j credentials from environment
    const uri = process.env.NEO4J_URI;
    const username = process.env.NEO4J_USER || process.env.NEO4J_USERNAME;
    const password = process.env.NEO4J_PASSWORD;
    
    // Initialize Neo4j connection
    if (service instanceof Neo4jService) {
      service.initialize(uri, username, password);
    }
  } else if (dbType === 'arangodb') {
    // Get ArangoDB credentials from environment
    const url = process.env.ARANGODB_URL;
    const dbName = process.env.ARANGODB_DB;
    const username = process.env.ARANGODB_USER;
    const password = process.env.ARANGODB_PASSWORD;
    
    // Initialize ArangoDB connection
    if (service instanceof ArangoDBService) {
      await service.initialize(url, dbName, username, password);
    }
  }
} 