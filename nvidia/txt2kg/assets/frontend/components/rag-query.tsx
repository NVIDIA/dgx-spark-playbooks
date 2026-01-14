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
import { useState, useEffect } from "react";
import { Triple } from "@/types/graph";
import { Search as SearchIcon, Zap, Database, Cpu } from "lucide-react";
import { LLMSelectorCompact } from "./llm-selector-compact";

interface RagQueryProps {
  onQuerySubmit: (query: string, params: RagParams) => Promise<void>;
  clearResults: () => void;
  isLoading: boolean;
  error?: string | null;
  className?: string;
  vectorEnabled?: boolean;
}

type QueryMode = 'traditional' | 'vector-search' | 'pure-rag';

export interface RagParams {
  kNeighbors: number;
  fanout: number;
  numHops: number;
  topK: number;
  useVectorSearch?: boolean;
  usePureRag?: boolean;
  queryMode?: QueryMode;
}

export function RagQuery({ 
  onQuerySubmit, 
  clearResults, 
  isLoading, 
  error,
  className,
  vectorEnabled 
}: RagQueryProps) {
  const [query, setQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [queryMode, setQueryMode] = useState<QueryMode>('traditional');
  const [params, setParams] = useState<RagParams>({
    kNeighbors: 4096,
    fanout: 400,
    numHops: 2,
    topK: 40,  // Increased to 40 for multi-hop paths (was 20, originally 5)
    useVectorSearch: false,
    usePureRag: false,
    queryMode: 'traditional'
  });

  // Update query mode when vectorEnabled changes - but don't override user selection
  useEffect(() => {
    // Only set default mode on initial load, not when user selects Graph Search
    if (vectorEnabled && queryMode === 'traditional') {
      console.log('Not forcing mode change - respecting user selection');
      // No longer forcing mode change - let user select what they want
    }
  }, [vectorEnabled, queryMode]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    // Set the appropriate params based on the selected query mode
    const updatedParams = { ...params };
    
    if (queryMode === 'pure-rag') {
      updatedParams.usePureRag = true;
      updatedParams.useVectorSearch = false;
    } else if (queryMode === 'vector-search') {
      updatedParams.usePureRag = false;
      updatedParams.useVectorSearch = true;
    } else {
      // traditional
      updatedParams.usePureRag = false;
      updatedParams.useVectorSearch = false;
    }
    
    // Pass the current queryMode to the parent component
    await onQuerySubmit(query, {
      ...updatedParams,
      queryMode: queryMode // Add the query mode to the params
    });
  };

  const handleReset = () => {
    setQuery("");
    clearResults();
  };

  const updateParam = (key: keyof RagParams, value: number | boolean) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  // Handle changing the query mode
  const handleQueryModeChange = (mode: QueryMode) => {
    console.log(`Changing query mode to: ${mode}`);
    setQueryMode(mode);
    
    // Force update params based on the selected mode
    const updatedParams = { ...params };
    if (mode === 'pure-rag') {
      updatedParams.usePureRag = true;
      updatedParams.useVectorSearch = false;
    } else if (mode === 'vector-search') {
      updatedParams.usePureRag = false;
      updatedParams.useVectorSearch = true;
    } else {
      // traditional mode
      updatedParams.usePureRag = false;
      updatedParams.useVectorSearch = false;
    }
    setParams(updatedParams);
  };

  return (
    <div className={`nvidia-build-card ${className}`}>
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-8 h-8 rounded-lg bg-nvidia-green/15 flex items-center justify-center">
            <SearchIcon className="h-4 w-4 text-nvidia-green" />
          </div>
          <h2 className="text-xl font-semibold text-foreground">RAG Query Engine</h2>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Query your knowledge graph using natural language
        </p>
      </div>
      
      {/* Query Type Selection */}
      <div className="mb-5">
        <h3 className="text-xs font-semibold text-foreground mb-3">Select Query Type</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <button
            type="button"
            onClick={() => handleQueryModeChange('pure-rag')}
            disabled={!vectorEnabled}
            className={`relative flex flex-col items-center p-3 border rounded-lg transition-all duration-200 hover:shadow-md ${
              queryMode === 'pure-rag' 
                ? 'border-nvidia-green bg-nvidia-green/10 text-nvidia-green shadow-sm' 
                : vectorEnabled 
                  ? 'border-border/40 hover:border-border/60 hover:bg-muted/20' 
                  : 'border-border/30 opacity-50 cursor-not-allowed'
            }`}
          >
            <div className={`w-5 h-5 rounded-md flex items-center justify-center mb-1.5 ${vectorEnabled ? 'bg-nvidia-green/15' : 'bg-muted/15'}`}>
              <Zap className={`h-2.5 w-2.5 ${vectorEnabled ? 'text-nvidia-green' : 'text-muted-foreground'}`} />
            </div>
            <span className={`text-sm font-semibold ${!vectorEnabled ? 'text-muted-foreground' : ''}`}>Pure RAG</span>
            <span className="text-[10px] mt-0.5 text-center text-muted-foreground leading-tight">
              Vector DB + LLM
            </span>
            {queryMode === 'pure-rag' && (
              <div className="absolute top-2 right-2 w-1.5 h-1.5 bg-nvidia-green rounded-full"></div>
            )}
            {!vectorEnabled && (
              <div className="text-[9px] px-1.5 py-0.5 bg-blue-500/20 text-blue-700 dark:text-blue-400 rounded mt-1 font-medium">
                NEEDS EMBEDDINGS
              </div>
            )}
          </button>

          <button
            type="button"
            onClick={() => handleQueryModeChange('traditional')}
            className={`relative flex flex-col items-center p-3 border rounded-lg transition-all duration-200 hover:shadow-md ${
              queryMode === 'traditional' 
                ? 'border-nvidia-green bg-nvidia-green/10 text-nvidia-green shadow-sm' 
                : 'border-border/40 hover:border-border/60 hover:bg-muted/20'
            }`}
          >
            <div className="w-5 h-5 rounded-md bg-nvidia-green/15 flex items-center justify-center mb-1.5">
              <Database className="h-2.5 w-2.5 text-nvidia-green" />
            </div>
            <span className="text-sm font-semibold">Graph Search</span>
            <span className="text-[10px] mt-0.5 text-center text-muted-foreground leading-tight">
              Graph DB + LLM
            </span>
            {queryMode === 'traditional' && (
              <div className="absolute top-2 right-2 w-1.5 h-1.5 bg-nvidia-green rounded-full"></div>
            )}
          </button>
          
          <button
            type="button"
            onClick={() => handleQueryModeChange('vector-search')}
            disabled={true}
            title="GraphRAG requires a GNN model (not yet available)"
            className="relative flex flex-col items-center p-3 border rounded-lg border-border/30 opacity-50 cursor-not-allowed"
          >
            <div className="w-5 h-5 rounded-md bg-muted/15 flex items-center justify-center mb-1.5">
              <Cpu className="h-2.5 w-2.5 text-muted-foreground" />
            </div>
            <span className="text-sm font-semibold text-muted-foreground">GraphRAG</span>
            <span className="text-[10px] mt-0.5 text-center text-muted-foreground leading-tight">
              RAG + GNN
            </span>
            <div className="text-[9px] px-1.5 py-0.5 bg-amber-500/20 text-amber-700 dark:text-amber-500 rounded mt-1 font-medium">
              COMING SOON
            </div>
          </button>
        </div>
      </div>
      
      {/* LLM Selection */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground mb-1">LLM Model</h3>
          <p className="text-xs text-muted-foreground">Select the LLM for answer generation</p>
        </div>
        <LLMSelectorCompact />
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="flex gap-3">
          <div className="relative flex-1">
            <input
              type="text"
              placeholder={
                queryMode === 'pure-rag' 
                  ? "Ask a question using RAG..." 
                  : queryMode === 'vector-search'
                    ? "Ask a question about your vector-enhanced knowledge graph..."
                    : "Ask a question about your knowledge graph..."
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full p-3 pl-11 rounded-xl border border-border/60 bg-background text-foreground focus:border-nvidia-green/50 focus:ring-2 focus:ring-nvidia-green/20 transition-colors"
              disabled={isLoading}
            />
            <SearchIcon className="absolute left-3.5 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          </div>
          <button 
            type="submit" 
            disabled={isLoading || !query.trim()} 
            className="px-6 py-3 bg-nvidia-green text-white hover:bg-nvidia-green/90 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:pointer-events-none shadow-sm"
          >
            {isLoading ? "Searching..." : "Search"}
          </button>
          <button
            type="button"
            onClick={handleReset}
            disabled={isLoading}
            className="px-4 py-3 border border-border/60 bg-background hover:bg-muted/30 text-foreground rounded-xl transition-colors disabled:opacity-50 disabled:pointer-events-none"
          >
            Clear
          </button>
        </div>

        {error && (
          <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-xl text-destructive text-sm">
            <div className="flex items-start gap-2">
              <div className="w-4 h-4 rounded-full bg-destructive/20 flex items-center justify-center mt-0.5 flex-shrink-0">
                <span className="text-xs">!</span>
              </div>
              <div>
                <strong>Error:</strong> {error}
              </div>
            </div>
          </div>
        )}

        <div>
          <button 
            type="button" 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-2 font-medium"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`h-4 w-4 transition-transform duration-200 ${showAdvanced ? 'rotate-180' : ''}`}>
              <path d="m6 9 6 6 6-6"/>
            </svg>
            {showAdvanced ? "Hide" : "Show"} Advanced Parameters
          </button>
          
          {showAdvanced && (
            <div className="mt-6 p-6 border border-border/40 rounded-xl bg-muted/10 space-y-6">
              {/* Only show graph-related parameters for traditional and vector search modes */}
              {queryMode !== 'pure-rag' && (
                <>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-foreground">KNN Neighbors</label>
                      <span className="text-sm font-semibold text-nvidia-green bg-nvidia-green/10 px-2 py-1 rounded-md">{params.kNeighbors}</span>
                    </div>
                    <input
                      type="range"
                      min="256"
                      max="8192"
                      step="256"
                      value={params.kNeighbors}
                      onChange={(e) => updateParam('kNeighbors', parseInt(e.target.value))}
                      className="w-full h-2 bg-muted/30 rounded-lg appearance-none cursor-pointer slider-thumb"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-foreground">Fanout</label>
                      <span className="text-sm font-semibold text-nvidia-green bg-nvidia-green/10 px-2 py-1 rounded-md">{params.fanout}</span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="1000"
                      step="50"
                      value={params.fanout}
                      onChange={(e) => updateParam('fanout', parseInt(e.target.value))}
                      className="w-full h-2 bg-muted/30 rounded-lg appearance-none cursor-pointer slider-thumb"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-foreground">Number of Hops</label>
                      <span className="text-sm font-semibold text-nvidia-green bg-nvidia-green/10 px-2 py-1 rounded-md">{params.numHops}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="4"
                      step="1"
                      value={params.numHops}
                      onChange={(e) => updateParam('numHops', parseInt(e.target.value))}
                      className="w-full h-2 bg-muted/30 rounded-lg appearance-none cursor-pointer slider-thumb"
                    />
                  </div>
                </>
              )}

              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-foreground">Top K Results</label>
                  <span className="text-sm font-semibold text-nvidia-green bg-nvidia-green/10 px-2 py-1 rounded-md">{params.topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="50"
                  step="1"
                  value={params.topK}
                  onChange={(e) => updateParam('topK', parseInt(e.target.value))}
                  className="w-full h-2 bg-muted/30 rounded-lg appearance-none cursor-pointer slider-thumb"
                />
              </div>
            </div>
          )}
        </div>
      </form>
    </div>
  );
} 