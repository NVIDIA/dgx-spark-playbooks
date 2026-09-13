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
"use client"

import { useState, useEffect } from "react"
import { useDocuments } from "@/contexts/document-context"
import { CheckCircle, Loader2, FileText, AlertCircle, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useRouter } from "next/navigation"
import { useShiftSelect } from "@/hooks/use-shift-select"

export function DocumentSelection() {
  const { documents, processDocuments, isProcessing } = useDocuments()
  const [processingStatus, setProcessingStatus] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [forceUpdate, setForceUpdate] = useState(0)
  const router = useRouter()

  // Use shift-select hook for document selection
  const {
    selectedItems: selectedDocs,
    setSelectedItems: setSelectedDocs,
    handleItemClick,
    handleSelectAll,
    isSelected
  } = useShiftSelect({
    items: documents,
    getItemId: (doc) => doc.id,
    canSelect: (doc) => doc.status === "New" || doc.status === "Processed" || doc.status === "Error",
    onSelectionChange: (selectedIds) => {
      // Optional: handle selection change if needed
    }
  })

  // Add event listener for document status changes
  useEffect(() => {
    const handleDocumentStatusChange = () => {
      console.log("Document status changed, forcing UI refresh");
      setForceUpdate(prev => prev + 1); // Increment to force a re-render
    };
    
    const handleProcessingComplete = () => {
      console.log("Processing complete event received, resetting UI state");
      setProcessingStatus(""); // Clear the processing status message
      setForceUpdate(prev => prev + 1); // Force a refresh
    };
    
    window.addEventListener('document-status-changed', handleDocumentStatusChange);
    window.addEventListener('processing-complete', handleProcessingComplete);
    
    return () => {
      window.removeEventListener('document-status-changed', handleDocumentStatusChange);
      window.removeEventListener('processing-complete', handleProcessingComplete);
    };
  }, []);
  
  // Add automatic UI refresh every second during processing
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isProcessing) {
      interval = setInterval(() => {
        setForceUpdate(prev => prev + 1); // Force UI refresh
      }, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isProcessing]);

  // On component mount, default to select all documents with status "New", "Processed", or "Error"
  useEffect(() => {
    const availableDocs = documents
      .filter(doc => doc.status === "New" || doc.status === "Processed" || doc.status === "Error")
      .map(doc => doc.id)
    
    setSelectedDocs(availableDocs)
  }, [documents, setSelectedDocs])

  const handleTabChange = (tab: string) => {
    const tabElement = document.querySelector(`[data-value="${tab}"]`)
    if (tabElement && 'click' in tabElement) {
      (tabElement as HTMLElement).click()
    }
  }

  const handleProcessDocuments = async () => {
    if (selectedDocs.length === 0) {
      setError("Please select at least one document to process")
      return
    }

    setError(null)
    setProcessingStatus("Preparing documents for processing...")

    try {
      // Update the processing status display
      const docNames = selectedDocs.map(id => 
        documents.find(d => d.id === id)?.name || 'Unknown'
      ).join(', ');
      
      setProcessingStatus(`Processing ${selectedDocs.length} document(s): ${docNames}`);
      
      // Call processDocuments with the selected document IDs
      await processDocuments(selectedDocs, {
        useLangChain: false,
        useGraphTransformer: false,
        promptConfigs: undefined
      })
      
      // Ensure UI is updated after processing completes
      setForceUpdate(prev => prev + 1);
      setProcessingStatus("Processing complete! Navigating to edit view...");
      
      // Short delay before navigation to allow status update to be seen
      setTimeout(() => {
        // Navigate to the edit tab after processing
        handleTabChange("edit")
      }, 1000);
    } catch (error) {
      console.error("Error processing documents:", error)
      setError("Failed to process documents. Please try again.")
    }
  }

  const handleStopProcessing = async () => {
    try {
      // Call the stop processing API
      const response = await fetch('/api/stop-processing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        setProcessingStatus("Processing stopped by user");
        setError(null);
        // Force UI refresh to update document statuses
        setForceUpdate(prev => prev + 1);
      } else {
        setError("Failed to stop processing. Please try again.");
      }
    } catch (error) {
      console.error("Error stopping processing:", error);
      setError("Failed to stop processing. Please try again.");
    }
  }

  return (
    <div className="space-y-4">
      <h3 className="text-md font-medium">Document Selection</h3>
      <p className="text-sm text-muted-foreground mb-2">Select which documents to process for triple extraction</p>
      
      {error && (
        <div className="bg-destructive/10 border border-destructive rounded-md p-4 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-destructive mt-0.5 flex-shrink-0" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      <div className="border rounded-md overflow-hidden">
        <div className="bg-muted/30 p-3 flex items-center justify-between">
          <div className="flex items-center">
            <input
              type="checkbox"
              className="rounded border-border text-primary focus:ring-primary mr-3 h-4 w-4"
              checked={selectedDocs.length === documents.filter(doc => doc.status === "New" || doc.status === "Processed" || doc.status === "Error").length && 
                      documents.filter(doc => doc.status === "New" || doc.status === "Processed" || doc.status === "Error").length > 0}
              onChange={handleSelectAll}
              disabled={documents.filter(doc => doc.status === "New" || doc.status === "Processed" || doc.status === "Error").length === 0 || isProcessing}
            />
            <span className="text-sm font-medium">
              {selectedDocs.length > 0 ? (
                <span className="text-nvidia-green text-xs">{selectedDocs.length} selected</span>
              ) : (
                <span className="text-xs">Select all</span>
              )}
            </span>
          </div>
          
          <div className="flex gap-2">
            <Button
              size="sm"
              onClick={handleProcessDocuments}
              disabled={selectedDocs.length === 0 || isProcessing}
              className="gap-1"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <FileText className="h-3.5 w-3.5" />
                  <span>Extract Triples</span>
                </>
              )}
            </Button>
            
            {isProcessing && (
              <Button
                size="sm"
                variant="destructive"
                onClick={handleStopProcessing}
                className="gap-1"
              >
                <X className="h-3.5 w-3.5" />
                <span>Stop</span>
              </Button>
            )}
          </div>
        </div>

        <div className="max-h-[200px] overflow-y-auto">
          {documents.length === 0 ? (
            <div className="p-6 text-center">
              <p className="text-muted-foreground">No documents available for processing</p>
              <Button 
                variant="link" 
                onClick={() => handleTabChange("upload")}
                className="mt-2"
              >
                Go to Upload
              </Button>
            </div>
          ) : (
            <table className="w-full text-sm">
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id} className="border-b last:border-b-0 hover:bg-muted/20">
                    <td className="pl-3 py-3">
                      <input
                        type="checkbox"
                        className="rounded border-border text-primary focus:ring-primary h-4 w-4"
                        checked={isSelected(doc.id)}
                        onChange={(e) => handleItemClick(doc, e)}
                        disabled={(doc.status !== "New" && doc.status !== "Processed" && doc.status !== "Error") || isProcessing}
                      />
                    </td>
                    <td className="px-3 py-3 font-medium text-foreground flex items-center gap-2 cursor-pointer" 
                        onClick={(e) => (doc.status === "New" || doc.status === "Processed" || doc.status === "Error") && !isProcessing && handleItemClick(doc, e)}>
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      {doc.name}
                    </td>
                    <td className="px-3 py-3">
                      <div className="flex items-center">
                        {doc.status === "New" && (
                          <span className="h-2 w-2 rounded-full bg-cyan-400 mr-2"></span>
                        )}
                        {doc.status === "Processing" && (
                          <Loader2 className="h-4 w-4 text-yellow-500 mr-2 animate-spin" />
                        )}
                        {doc.status === "Processed" && (
                          <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                        )}
                        {doc.status === "Error" && (
                          <AlertCircle className="h-4 w-4 text-destructive mr-2" />
                        )}
                        <span>{doc.status}</span>
                      </div>
                    </td>
                    <td className="px-3 py-3">{doc.size}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {isProcessing && processingStatus && (
        <div className="border rounded-md p-4 bg-primary/5">
          <div className="flex items-center gap-2 text-sm">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            <span>{processingStatus}</span>
          </div>
          <div className="mt-2 h-1 w-full bg-muted overflow-hidden rounded-full">
            <div className="h-full bg-primary rounded-full animate-progress"></div>
          </div>
        </div>
      )}
    </div>
  )
} 