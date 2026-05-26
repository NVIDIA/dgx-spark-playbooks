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

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertCircle, FileText, FileUp, Loader2, Zap } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/components/ui/use-toast";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useDocuments } from "@/contexts/document-context";

interface DocumentProcessorProps {
  onComplete?: (results: any) => void;
  className?: string;
}

export function DocumentProcessor({ onComplete, className }: DocumentProcessorProps) {
  const { addDocuments, processDocuments, documents } = useDocuments();
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [processingTab, setProcessingTab] = useState<string>("triples");
  const [useSentenceChunking, setUseSentenceChunking] = useState(true);
  const [useEntityExtraction, setUseEntityExtraction] = useState(true);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    if (selectedFile) {
      // Add file to document context for display in document list
      addDocuments([selectedFile]);
      setFile(selectedFile);
      setError(null);
      setProgress(0);
      setProcessingStatus("");
      
      // Show toast notification
      toast({
        title: "File Uploaded",
        description: `"${selectedFile.name}" added to document list.`,
        duration: 3000,
      });
    }
  };

  const processFile = async () => {
    if (!file) {
      setError("Please select a file to process");
      return;
    }

    try {
      setIsProcessing(true);
      setProgress(0);
      setProcessingStatus("Reading file...");
      setError(null);

      // Find the document ID for the file we're processing
      const docToProcess = documents.find(doc => doc.name === file.name);
      
      if (!docToProcess) {
        throw new Error("Document not found in document list");
      }

      // Use the document context to process documents with the specific ID
      await processDocuments([docToProcess.id], {
        useLangChain: true,
        useGraphTransformer: false,
        promptConfigs: undefined
      });
      setProgress(100);
      setProcessingStatus("Processing complete!");
      
      // Notify about completion
      toast({
        title: "Processing Complete",
        description: "Document has been processed successfully. You can now generate embeddings from the document table.",
        duration: 5000,
      });
      
      // Reset the file input
      setFile(null);
      
      // Call onComplete callback if provided
      if (onComplete) {
        onComplete({
          success: true,
          message: "Document processed successfully"
        });
      }
    } catch (err) {
      console.error("Error processing document:", err);
      setError(err instanceof Error ? err.message : "Unknown error processing document");
      
      toast({
        title: "Processing Failed",
        description: err instanceof Error ? err.message : "Failed to process document",
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Process triples from text
  const processTriples = async (text: string, filename: string) => {
    setProcessingStatus("Extracting triples with LangChain...");

    let triples;
    
    // If it's a CSV file, process each row as a document with LLM extraction
    if (filename.toLowerCase().endsWith('.csv')) {
      setProcessingStatus(`Processing CSV file rows as documents (${(text.length / 1024).toFixed(2)} KB)...`);
      try {
        console.log(`🔥 DocumentProcessor: Starting CSV row-by-row processing for file: ${filename}`);
        triples = await parseCSVToTriples(text);
        console.log(`🔥 DocumentProcessor: Extracted ${triples.length} triples from CSV file`);
        setProgress(60);
        
        // For very large triple sets, limit what we send to the API
        const maxTriplesToProcess = 10000;
        let triplesToProcess = triples;
        
        if (triples.length > maxTriplesToProcess) {
          console.log(`Limiting triples to ${maxTriplesToProcess} out of ${triples.length} total`);
          setProcessingStatus(`Processing ${maxTriplesToProcess} of ${triples.length} triples (limited for performance)...`);
          triplesToProcess = triples.slice(0, maxTriplesToProcess);
        }
        
        setProcessingStatus(`Processing ${triplesToProcess.length} triples...`);
        
        // Process document to create backend with embeddings
        // NOTE: This API no longer automatically stores triples in Neo4j.
        // Storage in Neo4j is now handled manually through the UI's "Store in Graph DB" button.
        const processingResponse = await fetch('/api/process-document', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            text: `CSV file with ${triples.length} triples`,  // Don't send the full CSV content
            filename,
            triples: triplesToProcess
          })
        });

        if (!processingResponse.ok) {
          throw new Error(`Failed to process document: ${processingResponse.statusText}`);
        }

        const processingData = await processingResponse.json();
        setProgress(100);
        setProcessingStatus("Processing complete!");

        // Notify about completion
        toast({
          title: "CSV Processing Complete",
          description: `Processed ${triplesToProcess.length} triples${triples.length > triplesToProcess.length ? ' (limited from ' + triples.length + ' total)' : ''} from your CSV file.`,
          duration: 5000,
        });

        // Call the onComplete callback with results
        if (onComplete) {
          onComplete({
            triples: triplesToProcess,
            totalTriples: triples.length,
            embeddings: processingData.embeddings || [],
            filename
          });
        }
        
        return; // Early return to skip the standard processing flow
      } catch (err) {
        console.error(`CSV processing error:`, err);
        throw new Error(`Failed to parse CSV file: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    
    // Standard processing for non-CSV files
    const extractResponse = await fetch('/api/extract-triples', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!extractResponse.ok) {
      throw new Error(`Failed to extract triples: ${extractResponse.statusText}`);
    }

    const extractData = await extractResponse.json();
    triples = extractData.triples;
    setProgress(60);
    
    setProcessingStatus("Generating embeddings...");

    // Process document to create backend with embeddings
    // NOTE: This API no longer automatically stores triples in Neo4j.
    // Storage in Neo4j is now handled manually through the UI's "Store in Graph DB" button.
    const processingResponse = await fetch('/api/process-document', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        text,
        filename,
        triples
      })
    });

    if (!processingResponse.ok) {
      throw new Error(`Failed to process document: ${processingResponse.statusText}`);
    }

    const processingData = await processingResponse.json();
    setProgress(100);
    setProcessingStatus("Processing complete!");

    // Notify about completion
    toast({
      title: "Triple Extraction Complete",
      description: `Extracted ${triples.length} triples and generated embeddings for the knowledge graph.`,
      duration: 5000,
    });

    // Call the onComplete callback with results
    if (onComplete) {
      onComplete({
        triples,
        embeddings: processingData.embeddings || [],
        filename
      });
    }
  };

  // Process sentence embeddings
  const processSentenceEmbeddings = async (text: string, filename: string) => {
    // If it's a CSV file, we need to convert it to text first
    let processableText = text;
    
    if (filename.toLowerCase().endsWith('.csv')) {
      setProcessingStatus("Preparing CSV data for embedding...");
      try {
        // For CSV files, we'll use the content of the cells as text to generate embeddings
        const triples = await parseCSVToTriples(text);
        // Create a text representation by joining subjects, predicates and objects
        processableText = triples
          .map(t => `${t.subject} ${t.predicate} ${t.object}`)
          .join('. ');
      } catch (err) {
        throw new Error(`Failed to process CSV file: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    
    setProcessingStatus("Chunking text into sentences...");

    // Call sentence embeddings API
    const embeddingsResponse = await fetch('/api/sentence-embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        text: processableText,
        documentId: filename 
      })
    });

    if (!embeddingsResponse.ok) {
      throw new Error(`Failed to process sentence embeddings: ${embeddingsResponse.statusText}`);
    }

    const embeddingsData = await embeddingsResponse.json();
    setProgress(100);
    setProcessingStatus("Sentence embeddings complete!");

    // Notify about completion
    toast({
      title: "Sentence Embeddings Complete",
      description: `Generated embeddings for ${embeddingsData.count} sentences from your document.`,
      duration: 5000,
    });

    // Show sample sentences in console for debugging
    console.log("Sample sentences:", embeddingsData.samples);

    // Call the onComplete callback with results
    if (onComplete) {
      onComplete({
        sentenceCount: embeddingsData.count,
        samples: embeddingsData.samples,
        filename
      });
    }
  };

  // Helper function to read file content
  const readFileContent = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      console.log(`Reading file: ${file.name}, size: ${(file.size / 1024).toFixed(2)} KB`);
      
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          const content = event.target.result as string;
          console.log(`File content loaded, length: ${content.length} characters`);
          
          // Special handling for CSV files
          if (file.name.toLowerCase().endsWith('.csv')) {
            try {
              console.log(`Processing CSV file content...`);
              // Don't parse here, just validate the content
              const lineCount = content.split('\n').length;
              console.log(`CSV file has ${lineCount} lines`);
              resolve(content);
            } catch (err) {
              console.error(`CSV parsing error:`, err);
              reject(new Error(`Failed to parse CSV file: ${err instanceof Error ? err.message : String(err)}`));
            }
          } else if (file.name.toLowerCase().endsWith('.json')) {
            try {
              console.log(`Processing JSON file content...`);
              // Convert JSON to readable text format for processing
              const textContent = convertJsonToText(content);
              console.log(`Converted JSON file to text format, length: ${textContent.length} characters`);
              resolve(textContent);
            } catch (err) {
              console.error(`JSON conversion error:`, err);
              reject(new Error(`Failed to process JSON file: ${err instanceof Error ? err.message : String(err)}`));
            }
          } else {
            resolve(content);
          }
        } else {
          reject(new Error("Failed to read file content"));
        }
      };
      reader.onerror = (error) => {
        console.error(`Error reading file:`, error);
        reject(new Error("Error reading file"));
      };
      reader.readAsText(file);
    });
  };

  // Helper function to convert JSON content to readable text format
  const convertJsonToText = (jsonContent: string): string => {
    try {
      // Parse the JSON to validate it
      const jsonData = JSON.parse(jsonContent);
      
      // Convert JSON to a readable text format that preserves structure and relationships
      const formatJsonObject = (obj: any, indent: number = 0): string => {
        const spaces = '  '.repeat(indent);
        
        if (obj === null || obj === undefined) {
          return 'null';
        }
        
        if (typeof obj === 'string' || typeof obj === 'number' || typeof obj === 'boolean') {
          return String(obj);
        }
        
        if (Array.isArray(obj)) {
          if (obj.length === 0) return '[]';
          const items = obj.map((item, index) => 
            `${spaces}  Item ${index + 1}: ${formatJsonObject(item, indent + 1)}`
          ).join('\n');
          return `[\n${items}\n${spaces}]`;
        }
        
        if (typeof obj === 'object') {
          const entries = Object.entries(obj);
          if (entries.length === 0) return '{}';
          
          const props = entries.map(([key, value]) => 
            `${spaces}  ${key}: ${formatJsonObject(value, indent + 1)}`
          ).join('\n');
          return `{\n${props}\n${spaces}}`;
        }
        
        return String(obj);
      };
      
      // Create a descriptive text representation
      let textContent = `JSON Document Content:\n\n`;
      textContent += formatJsonObject(jsonData);
      
      return textContent;
    } catch (error) {
      console.warn('Failed to parse JSON, treating as plain text:', error);
      // If JSON parsing fails, return the original content as-is
      return jsonContent;
    }
  }

  // Parse CSV file and process each row as a document for LLM-based triple extraction
  const parseCSVToTriples = async (csvContent: string): Promise<any[]> => {
    console.log(`Processing CSV content as individual documents, length: ${csvContent.length} characters`);
    
    // Split the CSV content into lines
    const lines = csvContent.split('\n').filter(line => line.trim().length > 0);
    console.log(`CSV has ${lines.length} non-empty lines`);
    
    if (lines.length < 2) {
      throw new Error("CSV file must contain a header row and at least one data row");
    }
    
    // Parse the header row
    const header = lines[0].split(',').map(h => h.trim().replace(/^"(.*)"$/, '$1'));
    console.log(`CSV headers: ${header.join(', ')}`);
    
    // Get data rows (skip header)
    const dataRows = lines.slice(1);
    console.log(`Processing ${dataRows.length} data rows as individual documents`);
    
    let allTriples: any[] = [];
    
    // Process each row as a separate document
    for (let rowIdx = 0; rowIdx < dataRows.length; rowIdx++) {
      const line = dataRows[rowIdx];
      setProcessingStatus(`Processing CSV row ${rowIdx + 1}/${dataRows.length} with LLM...`);
      
      try {
        // Parse CSV row into fields
        const fields: string[] = [];
        let fieldStart = 0;
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
          if (line[i] === '"') {
            inQuotes = !inQuotes;
          } else if (line[i] === ',' && !inQuotes) {
            fields.push(line.substring(fieldStart, i).trim().replace(/^"(.*)"$/, '$1'));
            fieldStart = i + 1;
          }
        }
        
        // Add the last field
        fields.push(line.substring(fieldStart).trim().replace(/^"(.*)"$/, '$1'));
        
        // Create document text from the row data
        let documentText = '';
        for (let i = 0; i < Math.min(header.length, fields.length); i++) {
          if (fields[i] && fields[i].trim()) {
            documentText += `${header[i]}: ${fields[i]}\n`;
          }
        }
        
        // Skip empty rows
        if (!documentText.trim()) {
          console.warn(`Skipping empty CSV row ${rowIdx + 1}`);
          continue;
        }
        
        console.log(`Processing row ${rowIdx + 1} as document: ${documentText.substring(0, 100)}...`);
        
        // Extract triples from this row's text using LLM
        try {
          const response = await fetch('/api/extract-triples', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              text: documentText,
              useLangChain: true // Use LLM-based extraction
            })
          });

          if (!response.ok) {
            console.error(`Failed to extract triples from row ${rowIdx + 1}: ${response.statusText}`);
            continue;
          }

          const data = await response.json();
          if (data.triples && Array.isArray(data.triples)) {
            console.log(`Extracted ${data.triples.length} triples from row ${rowIdx + 1}`);
            allTriples = allTriples.concat(data.triples);
          }
        } catch (error) {
          console.error(`Error processing row ${rowIdx + 1}:`, error);
          continue;
        }
        
        // Update progress
        setProgress(20 + (rowIdx / dataRows.length) * 40);
        
      } catch (parseError) {
        console.error(`Error parsing CSV row ${rowIdx + 1}:`, parseError);
        continue;
      }
    }
    
    console.log(`Successfully extracted ${allTriples.length} triples from ${dataRows.length} CSV rows`);
    return allTriples;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Process Document</CardTitle>
        <CardDescription>
          Extract triples from documents and build a knowledge graph
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-start gap-4">
            <div className="grid w-full gap-2">
              <label htmlFor="document-upload" className="cursor-pointer">
                <div className="flex h-24 w-full items-center justify-center rounded-md border border-dashed border-input bg-muted/50 p-4 hover:bg-muted/80 transition-colors">
                  <div className="flex flex-col items-center gap-2">
                    <FileUp className="h-10 w-10 text-muted-foreground" />
                    <span className="text-sm font-medium text-muted-foreground">
                      {file ? file.name : "Upload document"}
                    </span>
                  </div>
                </div>
                <input
                  id="document-upload"
                  type="file"
                  accept=".md,.txt,.csv"
                  onChange={handleFileChange}
                  className="sr-only"
                />
              </label>
            </div>
          </div>

          <Tabs 
            defaultValue="triples" 
            value={processingTab}
            onValueChange={setProcessingTab}
            className="w-full"
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="triples">Knowledge Triples</TabsTrigger>
              <TabsTrigger value="embeddings">Sentence Embeddings</TabsTrigger>
            </TabsList>
            <TabsContent value="triples">
              <div className="space-y-4 py-4">
                <div className="flex items-center space-x-2">
                  <Switch 
                    id="use-sentence-chunking" 
                    checked={useSentenceChunking}
                    onCheckedChange={setUseSentenceChunking}
                  />
                  <Label htmlFor="use-sentence-chunking">Use sentence-level chunking</Label>
                </div>
              </div>
            </TabsContent>
            <TabsContent value="embeddings">
              <div className="py-4 text-sm text-muted-foreground">
                You can now generate embeddings directly from the document table after processing.
                <div className="flex items-center mt-2 p-2 bg-muted/30 rounded-md">
                  <Zap className="h-4 w-4 text-primary mr-2" />
                  <span>Click the lightning icon in the document table to generate embeddings</span>
                </div>
              </div>
            </TabsContent>
          </Tabs>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {isProcessing && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>{processingStatus}</span>
              </div>
              <Progress value={progress} className="h-2 w-full" />
            </div>
          )}

          <Button
            onClick={processFile}
            className="w-full"
            disabled={!file || isProcessing}
          >
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <FileText className="mr-2 h-4 w-4" />
                Process Document & Generate Triples
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
} 