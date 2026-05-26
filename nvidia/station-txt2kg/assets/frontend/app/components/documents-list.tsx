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
"use client";

import { useState, useEffect } from "react";
import { Triple } from "@/types/graph";

export default function DocumentsList() {
  const [loading, setLoading] = useState(true);
  const [triples, setTriples] = useState<Triple[]>([]);
  const [entities, setEntities] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchTriplesAndEntities() {
      try {
        setLoading(true);
        
        // Fetch triples from Neo4j
        const response = await fetch('/api/neo4j/triples', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        });
        
        if (!response.ok) {
          throw new Error(`Failed to fetch triples: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Extract unique entities
        const uniqueEntities = new Set<string>();
        data.triples.forEach((triple: Triple) => {
          uniqueEntities.add(triple.subject);
          uniqueEntities.add(triple.object);
        });
        
        // Store data in local storage, overwriting previous data
        localStorage.setItem("graphTriples", JSON.stringify(data.triples));
        localStorage.setItem("graphDocumentName", "sample_data.csv");
        
        // Update state
        setTriples(data.triples);
        setEntities(Array.from(uniqueEntities));
        setError(null);
      } catch (err) {
        console.error("Error fetching data:", err);
        setError(err instanceof Error ? err.message : "Unknown error occurred");
      } finally {
        setLoading(false);
      }
    }
    
    fetchTriplesAndEntities();
  }, []);
  
  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Document Data</h2>
      
      {loading && (
        <div className="text-center py-8">
          <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto"></div>
          <p className="mt-2">Loading data from Neo4j...</p>
        </div>
      )}
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}
      
      {!loading && !error && (
        <div>
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2">Entities ({entities.length})</h3>
            <div className="border rounded-md p-4 bg-background max-h-64 overflow-y-auto">
              <ul className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {entities.slice(0, 100).map((entity, index) => (
                  <li key={index} className="text-sm truncate">{entity}</li>
                ))}
              </ul>
              {entities.length > 100 && (
                <p className="text-sm text-muted-foreground mt-2">
                  Showing 100 of {entities.length} entities
                </p>
              )}
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">Triples ({triples.length})</h3>
            <div className="border rounded-md overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-muted/50 border-b border-border">
                      <th className="px-4 py-3 text-left text-sm font-semibold text-muted-foreground">Subject</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-muted-foreground">Predicate</th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-muted-foreground">Object</th>
                    </tr>
                  </thead>
                  <tbody>
                    {triples.slice(0, 50).map((triple, index) => (
                      <tr key={index} className="border-t border-border hover:bg-muted/30 transition-colors">
                        <td className="px-4 py-2 text-sm text-foreground">{triple.subject}</td>
                        <td className="px-4 py-2 text-sm text-foreground">{triple.predicate}</td>
                        <td className="px-4 py-2 text-sm text-foreground">{triple.object}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {triples.length > 50 && (
                <p className="text-sm text-muted-foreground p-4 border-t">
                  Showing 50 of {triples.length} triples
                </p>
              )}
            </div>
          </div>
          
          <div className="mt-6">
            <button
              onClick={() => {
                localStorage.setItem("graphTriples", JSON.stringify(triples));
                localStorage.setItem("graphDocumentName", "sample_data.csv");
                alert(`Saved ${triples.length} triples to local storage. You can now view the graph visualization.`);
              }}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
            >
              Save to Local Storage
            </button>
          </div>
        </div>
      )}
    </div>
  );
} 