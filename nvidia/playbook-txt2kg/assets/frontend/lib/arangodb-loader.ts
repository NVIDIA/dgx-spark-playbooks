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
import { ArangoDBService } from './arangodb';

/**
 * Load triples from ArangoDB for use with the TXT2KG dataset
 * @param url ArangoDB connection URL
 * @param dbName ArangoDB database name
 * @param username ArangoDB username
 * @param password ArangoDB password 
 * @returns Array of triples in the format expected by create_remote_backend_from_triplets
 */
export async function loadTriplesFromArangoDB(
  url?: string,
  dbName?: string,
  username?: string,
  password?: string
): Promise<string[]> {
  try {
    // Get the ArangoDB service instance
    const arangoService = ArangoDBService.getInstance();
    
    // Initialize the connection if not already initialized
    if (!arangoService.isInitialized()) {
      await arangoService.initialize(url, dbName, username, password);
    }
    
    // Query to get all triples from ArangoDB
    const query = `
      FOR e IN relationships
      LET subject = DOCUMENT(e._from).name
      LET object = DOCUMENT(e._to).name
      LET predicate = e.type
      RETURN subject + " " + predicate + " " + object
    `;
    
    // Execute the query
    const results = await arangoService.executeQuery(query);
    
    // Format the triples as strings in the format "subject predicate object"
    const triples = results.map(triple => triple);
    
    console.log(`Loaded ${triples.length} triples from ArangoDB`);
    return triples;
  } catch (error) {
    console.error('Error loading triples from ArangoDB:', error);
    throw error;
  }
} 