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

import { useState } from "react"
import { CheckCircle, AlertCircle, Loader2, Trash2, FileText, Table, Edit, Eye, Network, Download, Info } from "lucide-react"
import { useDocuments } from "@/contexts/document-context"
import { DocumentActions } from "@/components/document-actions"
import { useShiftSelect } from "@/hooks/use-shift-select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import type { Triple } from "@/utils/text-processing"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { downloadDocument } from "@/lib/utils"
import { toast } from "@/hooks/use-toast"

export interface DocumentsTableProps {
  onTabChange?: (tab: string) => void;
}

export function DocumentsTable({ onTabChange }: DocumentsTableProps) {
  const { documents, deleteDocuments, updateTriples } = useDocuments()
  const [showTriplesDialog, setShowTriplesDialog] = useState(false)
  const [currentDocumentId, setCurrentDocumentId] = useState<string | null>(null)
  const [editableTriples, setEditableTriples] = useState<Triple[]>([])
  const [editingTripleIndex, setEditingTripleIndex] = useState<number | null>(null)
  
  // Delete confirmation dialog state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<{ type: 'single' | 'multiple', docId?: string, docName?: string } | null>(null)

  // Use shift-select hook for document selection
  const {
    selectedItems: selectedDocuments,
    setSelectedItems: setSelectedDocuments,
    handleItemClick,
    handleSelectAll,
    isSelected
  } = useShiftSelect({
    items: documents,
    getItemId: (doc) => doc.id,
    canSelect: () => true, // All documents can be selected in this table
    onSelectionChange: (selectedIds) => {
      // Optional: handle selection change if needed
    }
  })

  const handleDeleteSelected = () => {
    if (selectedDocuments.length === 0) return
    setDeleteTarget({ type: 'multiple' })
    setShowDeleteDialog(true)
  }
  
  const handleConfirmDelete = () => {
    if (!deleteTarget) return
    
    if (deleteTarget.type === 'multiple') {
      deleteDocuments(selectedDocuments)
      setSelectedDocuments([])
      toast({
        title: "Documents Deleted",
        description: `Successfully deleted ${selectedDocuments.length} document(s).`,
        duration: 3000,
      })
    } else if (deleteTarget.type === 'single' && deleteTarget.docId) {
      deleteDocuments([deleteTarget.docId])
      toast({
        title: "Document Deleted",
        description: `"${deleteTarget.docName}" has been deleted.`,
        duration: 3000,
      })
    }
    
    setShowDeleteDialog(false)
    setDeleteTarget(null)
  }
  
  const openTriplesDialog = (documentId: string) => {
    const document = documents.find(doc => doc.id === documentId);
    if (document && document.triples) {
      setCurrentDocumentId(documentId);
      setEditableTriples([...document.triples]);
      setShowTriplesDialog(true);
    }
  }
  
  const saveTriples = () => {
    if (currentDocumentId) {
      updateTriples(currentDocumentId, editableTriples);
      setShowTriplesDialog(false);
    }
  }
  
  const updateTriple = (index: number, field: 'subject' | 'predicate' | 'object', value: string) => {
    const newTriples = [...editableTriples];
    newTriples[index] = {
      ...newTriples[index],
      [field]: value
    };
    setEditableTriples(newTriples);
  }
  
  const deleteTriple = (index: number) => {
    const newTriples = [...editableTriples];
    newTriples.splice(index, 1);
    setEditableTriples(newTriples);
  }
  
  const addNewTriple = () => {
    setEditableTriples([...editableTriples, { subject: '', predicate: '', object: '' }]);
    setEditingTripleIndex(editableTriples.length);
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "New":
        return <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 mr-2"></span>
      case "Processing":
        return <Loader2 className="h-3.5 w-3.5 text-yellow-500 mr-2 animate-spin" />
      case "Processed":
        return <CheckCircle className="h-3.5 w-3.5 text-green-500 mr-2" />
      case "Error":
        return <AlertCircle className="h-3.5 w-3.5 text-destructive mr-2" />
      default:
        return <span className="h-1.5 w-1.5 rounded-full bg-gray-400 mr-2"></span>
    }
  }

  // Show different columns based on document processing state
  const showTriplesColumn = documents.some(doc => doc.status === 'Processed')



  return (
    <div className="relative">
      <div className="flex justify-between items-center p-6 bg-muted/10 border-b border-border/20">
        <div className="flex items-center">
          <div className="relative flex items-center">
            <input
              type="checkbox"
              className="rounded border-border selection-accent mr-4 h-4 w-4"
              checked={selectedDocuments.length === documents.length && documents.length > 0}
              onChange={handleSelectAll}
              disabled={documents.length === 0}
            />
            <div className="flex flex-col">
              <span className="text-sm font-medium">
                {selectedDocuments.length > 0 ? (
                  <span className="text-nvidia-green">{selectedDocuments.length} selected</span>
                ) : (
                  <span className="text-foreground">Select all documents</span>
                )}
              </span>
              {documents.length > 0 && (
                <span className="text-xs text-muted-foreground">
                  {documents.length} total document{documents.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {selectedDocuments.length > 0 && (
            <button 
              onClick={handleDeleteSelected} 
              className="flex items-center gap-2 px-3 py-2 text-sm font-medium bg-red-500/10 hover:bg-red-500/20 text-red-600 dark:text-red-400 rounded-lg transition-colors"
            >
              <Trash2 className="h-4 w-4" />
              <span>Delete Selected ({selectedDocuments.length})</span>
            </button>
          )}
        </div>
      </div>

      <div className="overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/20 bg-muted/5">
              <th className="w-12 pl-6 py-3"></th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-left py-3">Name</th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-left py-3">Status</th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-left py-3">Upload Status</th>
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-right py-3 pr-4">Size (KB)</th>
              {showTriplesColumn && <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center py-3">Triples</th>}
              <th className="text-xs font-semibold text-muted-foreground uppercase tracking-wider text-center py-3 pr-6">Actions</th>
            </tr>
          </thead>
        <tbody>
          {documents.length === 0 ? (
            <tr>
              <td colSpan={showTriplesColumn ? 7 : 6} className="py-16">
                <div className="flex flex-col items-center justify-center text-center">
                  <div className="w-24 h-24 rounded-2xl bg-nvidia-green/10 flex items-center justify-center mb-6 border-2 border-dashed border-nvidia-green/20">
                    <FileText className="h-12 w-12 text-nvidia-green" />
                  </div>
                  <h3 className="text-xl font-semibold text-foreground mb-3">No documents uploaded yet</h3>
                  <p className="text-sm text-muted-foreground mb-6 max-w-md leading-relaxed">
                    Get started by uploading markdown, CSV, text, or JSON files to extract knowledge graphs
                  </p>
                  <div className="inline-flex items-center gap-2 text-xs text-muted-foreground bg-muted/40 px-3 py-1.5 rounded-full border border-border/30">
                    <Info className="h-4 w-4" />
                    <span>Supported: .md, .csv, .txt, .json</span>
                  </div>
                </div>
              </td>
            </tr>
          ) : (
            documents.map((doc) => (
               <tr key={doc.id} className={`transition-all duration-200 hover:bg-nvidia-green/5 cursor-pointer group border-b border-border/10 last:border-b-0 ${isSelected(doc.id) ? 'bg-nvidia-green/8 border-l-4 border-l-nvidia-green' : 'hover:border-l-4 hover:border-l-nvidia-green/40'}`}
                   onClick={(e) => handleItemClick(doc, e)}>
                <td className="pl-6 py-4" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    className="rounded border-border selection-accent h-4 w-4"
                    checked={isSelected(doc.id)}
                    onChange={(e) => handleItemClick(doc, e)}
                  />
                </td>
                 <td className="py-4">
                  <div className="flex items-center gap-3">
                    <FileText className="h-4 w-4 text-nvidia-green flex-shrink-0" />
                    <span className="text-sm font-medium text-foreground truncate max-w-[200px]" title={doc.name}>{doc.name}</span>
                  </div>
                </td>
                <td className="py-4">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(doc.status)}
                    <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                      doc.status === 'Processed' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                      doc.status === 'Processing' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400' :
                      doc.status === 'Error' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                      'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-400'
                    }`}>{doc.status}</span>
                  </div>
                </td>
                <td className="py-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-nvidia-green" />
                    <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">{doc.uploadStatus}</span>
                  </div>
                </td>
                <td className="py-4 text-right pr-4">
                  <span className="text-xs font-mono bg-muted/50 px-2 py-1 rounded">{doc.size}</span>
                </td>
                {showTriplesColumn && (
                  <td className="py-4 text-center">
                    {doc.status === "Processed" && doc.triples ? (
                      <div className="flex items-center justify-center gap-3">
                        <span className="text-xs font-bold text-nvidia-green bg-nvidia-green/15 px-2.5 py-1 rounded-full">{doc.triples.length}</span>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            openTriplesDialog(doc.id);
                          }}
                          className="p-2 text-nvidia-green hover:bg-nvidia-green/10 rounded-lg transition-colors"
                          aria-label={`View and edit ${doc.triples?.length || 0} triples for ${doc.name}`}
                          title="View and edit triples"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                      </div>
                    ) : doc.status === "Error" ? (
                      <span className="text-xs text-destructive font-medium">Error</span>
                    ) : (
                      <span className="text-xs text-muted-foreground">-</span>
                    )}
                  </td>
                )}
                <td className="py-4 pr-6">
                  <div className="flex items-center justify-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button 
                      onClick={(e) => {
                        e.stopPropagation()
                        // Create a simple info modal or tooltip showing document details
                      }}
                      className="p-2 text-muted-foreground hover:text-nvidia-green hover:bg-nvidia-green/10 rounded-lg transition-colors"
                      aria-label={`View info for ${doc.name}`}
                      title="View document info"
                    >
                      <Info className="h-4 w-4" />
                    </button>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation()
                        try {
                          downloadDocument(doc.file, doc.name)
                          toast({
                            title: "Download Started",
                            description: `"${doc.name}" is being downloaded.`,
                            duration: 3000,
                          })
                        } catch (error) {
                          console.error('Download failed:', error)
                          toast({
                            title: "Download Failed",
                            description: `Failed to download "${doc.name}". Please try again.`,
                            variant: "destructive",
                            duration: 5000,
                          })
                        }
                      }}
                      className="p-2 text-muted-foreground hover:text-nvidia-green hover:bg-nvidia-green/10 rounded-lg transition-colors"
                      aria-label={`Download ${doc.name}`}
                      title="Download document"
                    >
                      <Download className="h-4 w-4" />
                    </button>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation()
                        setDeleteTarget({ type: 'single', docId: doc.id, docName: doc.name })
                        setShowDeleteDialog(true)
                      }}
                      className="p-2 text-muted-foreground hover:text-red-500 hover:bg-red-500/10 rounded-lg transition-colors"
                      aria-label={`Delete ${doc.name}`}
                      title="Delete document"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </td>
              </tr>
            ))
          )}
        </tbody>
        </table>
      </div>
      
      <Dialog open={showTriplesDialog} onOpenChange={setShowTriplesDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="nvidia-build-h3">Edit Knowledge Graph Triples</DialogTitle>
            <DialogDescription className="nvidia-build-body text-muted-foreground">
              View and edit the extracted triples before processing into your graph database
            </DialogDescription>
          </DialogHeader>
          
          <div className="mt-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <span className="nvidia-build-h3">{editableTriples.length} Triples</span>
                <p className="nvidia-build-caption text-muted-foreground mt-1">Subject-Predicate-Object relationships</p>
              </div>
              <Button variant="outline" size="sm" onClick={addNewTriple} className="nvidia-build-button">
                <Edit className="h-4 w-4 mr-2" />
                Add Triple
              </Button>
            </div>
            
            <div className="border rounded-md overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="bg-muted/50 border-b border-border">
                    <th className="px-4 py-3 text-left text-sm font-semibold text-muted-foreground">Subject</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-muted-foreground">Predicate</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-muted-foreground">Object</th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-muted-foreground w-20">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {editableTriples.map((triple, index) => (
                    <tr key={index} className="border-b border-border last:border-b-0 hover:bg-muted/30 transition-colors">
                      <td className="px-4 py-2">
                        {editingTripleIndex === index ? (
                          <input
                            type="text"
                            value={triple.subject}
                            onChange={(e) => updateTriple(index, 'subject', e.target.value)}
                            className="w-full bg-background border border-input rounded p-2 text-sm text-foreground focus:ring-2 focus:ring-primary/50 focus:border-primary"
                          />
                        ) : (
                          <span className="text-sm text-foreground">{triple.subject}</span>
                        )}
                      </td>
                      <td className="px-4 py-2">
                        {editingTripleIndex === index ? (
                          <input
                            type="text"
                            value={triple.predicate}
                            onChange={(e) => updateTriple(index, 'predicate', e.target.value)}
                            className="w-full bg-background border border-input rounded p-2 text-sm text-foreground focus:ring-2 focus:ring-primary/50 focus:border-primary"
                          />
                        ) : (
                          <span className="text-sm text-foreground">{triple.predicate}</span>
                        )}
                      </td>
                      <td className="px-4 py-2">
                        {editingTripleIndex === index ? (
                          <input
                            type="text"
                            value={triple.object}
                            onChange={(e) => updateTriple(index, 'object', e.target.value)}
                            className="w-full bg-background border border-input rounded p-2 text-sm text-foreground focus:ring-2 focus:ring-primary/50 focus:border-primary"
                          />
                        ) : (
                          <span className="text-sm text-foreground">{triple.object}</span>
                        )}
                      </td>
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-1">
                          {editingTripleIndex === index ? (
                            <button
                              onClick={() => setEditingTripleIndex(null)}
                              className="p-1.5 text-primary hover:text-primary/80 hover:bg-primary/10 rounded-full transition-colors"
                              aria-label={`Save changes to triple: ${triple.subject} ${triple.predicate} ${triple.object}`}
                              title="Save"
                            >
                              <CheckCircle className="h-4 w-4" />
                            </button>
                          ) : (
                            <button
                              onClick={() => setEditingTripleIndex(index)}
                              className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-full transition-colors"
                              aria-label={`Edit triple: ${triple.subject} ${triple.predicate} ${triple.object}`}
                              title="Edit"
                            >
                              <Edit className="h-4 w-4" />
                            </button>
                          )}
                          <button
                            onClick={() => deleteTriple(index)}
                            className="p-1.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-full transition-colors"
                            aria-label={`Delete triple: ${triple.subject} ${triple.predicate} ${triple.object}`}
                            title="Delete"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="flex justify-end mt-4">
              <Button onClick={saveTriples}>
                Save Changes
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
      
      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <Trash2 className="h-5 w-5 text-destructive" />
              Delete {deleteTarget?.type === 'multiple' ? 'Documents' : 'Document'}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {deleteTarget?.type === 'multiple' ? (
                <>
                  Are you sure you want to delete <strong>{selectedDocuments.length}</strong> selected document{selectedDocuments.length !== 1 ? 's' : ''}? 
                  This action cannot be undone.
                </>
              ) : (
                <>
                  Are you sure you want to delete <strong>"{deleteTarget?.docName}"</strong>? 
                  This action cannot be undone.
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setDeleteTarget(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleConfirmDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

