//
// SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
"use client"

import { useState, useEffect } from "react"
import { AlertCircle, Loader2 } from "lucide-react"

const POLL_INTERVAL_MS = 60_000

export function BackendReadinessBanner() {
  const [showBanner, setShowBanner] = useState(false)

  useEffect(() => {
    let mounted = true

    const check = async () => {
      let isVllmMode = false
      try {
        const configRes = await fetch("/api/config")
        if (!mounted || !configRes.ok) return
        const config = await configRes.json()
        const mode = config.backendMode ?? "ollama"
        isVllmMode = mode === "vllm"

        if (mode !== "vllm") {
          setShowBanner(false)
          return
        }

        const vllmRes = await fetch("/api/vllm/models", {
          signal: AbortSignal.timeout(5000),
        })
        if (!mounted) return
        const data = vllmRes.ok ? await vllmRes.json() : { models: [] }
        const hasModels = Array.isArray(data.models) && data.models.length > 0
        setShowBanner(!hasModels)
      } catch {
        if (mounted && isVllmMode) setShowBanner(true)
      }
    }

    check()
    const id = setInterval(check, POLL_INTERVAL_MS)
    return () => {
      mounted = false
      clearInterval(id)
    }
  }, [])

  if (!showBanner) return null

  return (
    <div
      className="flex items-center gap-3 px-4 py-3 bg-amber-500/15 border-b border-amber-500/30 text-amber-800 dark:text-amber-200"
      role="status"
      aria-live="polite"
    >
      <Loader2 className="h-5 w-5 shrink-0 animate-spin text-amber-600 dark:text-amber-400" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium">
          Backend (vLLM) is still initializing. This can take 30+ minutes after start.
        </p>
        <p className="text-xs mt-0.5 opacity-90">
          Check progress: <code className="bg-black/10 dark:bg-white/10 px-1 rounded">docker logs vllm-service -f</code>
        </p>
      </div>
      <AlertCircle className="h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" aria-hidden />
    </div>
  )
}
