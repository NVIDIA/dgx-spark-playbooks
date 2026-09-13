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

import { useState, useEffect } from "react"
import { Key, Lock, ArrowRight, X } from "lucide-react"

// DEPRECATED: This component was used for xAI API key management.
// xAI integration has been removed, so this component is now non-functional.
// It remains for backward compatibility only.
export function ApiKeyPrompt() {
  const [apiKey, setApiKey] = useState("")
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    // API key prompt is completely disabled - xAI integration removed
    setIsVisible(false)
  }, [])

  // Close on Escape
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsVisible(false)
    }
    document.addEventListener("keydown", onKeyDown)
    return () => document.removeEventListener("keydown", onKeyDown)
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!apiKey.trim()) return

    // xAI integration has been removed - this function is deprecated
    console.log("API Key prompt is deprecated - xAI integration removed")
    setIsVisible(false)
  }

  // Public function to show the modal again (can be called from other components)
  const showPrompt = () => {
    // API key prompt is disabled - xAI integration has been removed
    console.log("API key prompt is disabled - xAI integration has been removed.")
    return false
  }

  // Attach the function to the window object for backward compatibility
  useEffect(() => {
    // @ts-ignore
    window.showApiKeyPrompt = showPrompt
    return () => {
      // @ts-ignore
      delete window.showApiKeyPrompt
    }
  }, [])

  if (!isVisible) return null

  return (
    <div
      className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50"
      onClick={() => setIsVisible(false)}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="glass-card rounded-xl p-8 max-w-md w-full mx-4 relative"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          aria-label="Close"
          className="absolute top-3 right-3 p-2 rounded-md hover:bg-muted/30 text-muted-foreground hover:text-foreground"
          onClick={() => setIsVisible(false)}
        >
          <X className="h-4 w-4" />
        </button>
        <div className="flex items-center gap-4 mb-6">
          <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center">
            <Key className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-bold">API Key (Deprecated)</h2>
            <p className="text-muted-foreground text-sm">xAI integration has been removed</p>
          </div>
        </div>

        <p className="text-foreground mb-6">
          xAI integration has been removed. This prompt is no longer functional.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Lock className="h-5 w-5 text-muted-foreground" />
            </div>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="deprecated"
              className="w-full bg-background border border-border rounded-lg p-3 pl-10 text-foreground focus:ring-2 focus:ring-primary/50 focus:border-primary transition-colors"
              required
            />
          </div>

          <div className="flex justify-end">
            <button type="submit" className="btn-primary">
              <span>Submit</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

