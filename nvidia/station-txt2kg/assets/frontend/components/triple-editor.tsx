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
import { Check, X } from "lucide-react"

interface TripleEditorProps {
  triple?: Triple
  index?: number
  onSave: (triple: Triple, index?: number) => void
  onCancel: () => void
}

export function TripleEditor({ triple, index, onSave, onCancel }: TripleEditorProps) {
  const [subject, setSubject] = useState(triple?.subject || "")
  const [predicate, setPredicate] = useState(triple?.predicate || "")
  const [object, setObject] = useState(triple?.object || "")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (subject.trim() && predicate.trim() && object.trim()) {
      onSave({ subject, predicate, object }, index)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-muted/30 border-b border-border">
      <div className="grid grid-cols-3 gap-4 mb-3">
        <div>
          <label htmlFor="subject" className="block text-xs text-muted-foreground mb-1">
            Subject
          </label>
          <input
            id="subject"
            type="text"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            className="w-full bg-background border border-border rounded-md p-2 text-sm text-foreground focus:ring-2 focus:ring-primary/50 focus:border-primary"
            placeholder="Entity"
            required
          />
        </div>
        <div>
          <label htmlFor="predicate" className="block text-xs text-muted-foreground mb-1">
            Predicate
          </label>
          <input
            id="predicate"
            type="text"
            value={predicate}
            onChange={(e) => setPredicate(e.target.value)}
            className="w-full bg-background border border-border rounded-md p-2 text-sm text-foreground focus:ring-2 focus:ring-primary/50 focus:border-primary"
            placeholder="Relation"
            required
          />
        </div>
        <div>
          <label htmlFor="object" className="block text-xs text-muted-foreground mb-1">
            Object
          </label>
          <input
            id="object"
            type="text"
            value={object}
            onChange={(e) => setObject(e.target.value)}
            className="w-full bg-background border border-border rounded-md p-2 text-sm text-foreground focus:ring-2 focus:ring-primary/50 focus:border-primary"
            placeholder="Entity"
            required
          />
        </div>
      </div>
      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          aria-label="Cancel editing triple"
          className="p-2 text-muted-foreground hover:text-foreground rounded-full hover:bg-muted/50 transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
        <button 
          type="submit" 
          aria-label="Save triple"
          className="p-2 text-primary hover:text-primary/80 rounded-full hover:bg-primary/10 transition-colors"
        >
          <Check className="h-4 w-4" />
        </button>
      </div>
    </form>
  )
}

