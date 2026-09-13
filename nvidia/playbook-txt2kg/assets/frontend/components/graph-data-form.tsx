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

import type React from "react"

import { useState } from "react"
import type { Triple } from "@/utils/text-processing"
import { GraphVisualization } from "./graph-visualization"

export function GraphDataForm() {
  const [triples, setTriples] = useState<Triple[]>([])
  const [documentName, setDocumentName] = useState("")
  const [dataSubmitted, setDataSubmitted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [jsonInput, setJsonInput] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    try {
      if (!jsonInput.trim()) {
        throw new Error("Please enter JSON data")
      }

      const parsedTriples = JSON.parse(jsonInput)

      if (!Array.isArray(parsedTriples)) {
        throw new Error("Invalid data format. Expected an array of triples.")
      }

      // Validate that each item has the required fields
      for (const triple of parsedTriples) {
        if (!triple.subject || !triple.predicate || !triple.object) {
          throw new Error("Each triple must have 'subject', 'predicate', and 'object' properties.")
        }
      }

      // Store in localStorage for persistence
      localStorage.setItem("graphTriples", jsonInput)
      if (documentName) {
        localStorage.setItem("graphDocumentName", documentName)
      }

      setTriples(parsedTriples)
      setDataSubmitted(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to parse JSON data")
    }
  }

  const handleJsonChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setJsonInput(e.target.value)
  }

  if (dataSubmitted) {
    return (
      <div className="h-[calc(100vh-400px)]">
        <h2 className="text-xl font-bold mb-4">
          Knowledge Graph: <span className="text-primary">{documentName || "Custom Data"}</span>
        </h2>
        <GraphVisualization triples={triples} fullscreen />
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto">
      <p className="text-muted-foreground mb-4">
        If automatic data loading failed, you can paste your triples data here in JSON format.
      </p>

      {error && (
        <div className="bg-destructive/10 border border-destructive rounded-lg p-3 mb-4">
          <p className="text-destructive">{error}</p>
        </div>
      )}

      <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mb-4">
        <h3 className="text-sm font-medium text-primary mb-2">Expected JSON Format</h3>
        <p className="text-xs text-muted-foreground mb-2">
          The data should be an array of objects, each with "subject", "predicate", and "object" properties:
        </p>
        <pre className="text-xs bg-card p-3 rounded overflow-auto">
          {`[
  {
    "subject": "NVIDIA",
    "predicate": "develops",
    "object": "GPUs"
  },
  {
    "subject": "GPUs",
    "predicate": "used for",
    "object": "AI training"
  }
]`}
        </pre>
        <p className="text-xs text-muted-foreground mt-2">
          You can export this format directly from the main application using the "Export as JSON" option.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="document-name" className="block text-sm font-medium text-foreground mb-2">
            Document Name (optional)
          </label>
          <input
            type="text"
            id="document-name"
            value={documentName}
            onChange={(e) => setDocumentName(e.target.value)}
            className="w-full bg-background border border-border rounded-lg p-3 text-foreground"
            placeholder="My Document"
          />
        </div>

        <div>
          <label htmlFor="triples-data" className="block text-sm font-medium text-foreground mb-2">
            Triples Data (JSON format)
          </label>
          <textarea
            id="triples-data"
            value={jsonInput}
            onChange={handleJsonChange}
            className="w-full h-64 bg-background border border-border rounded-lg p-3 text-foreground font-mono text-sm"
            placeholder='[{"subject":"NVIDIA","predicate":"develops","object":"GPUs"}]'
            required
          ></textarea>
        </div>

        <button type="submit" className="btn-primary">
          Visualize Graph
        </button>
      </form>
    </div>
  )
}

