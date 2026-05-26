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

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Info, Sparkles, Eye, Upload, Zap } from "lucide-react"

export function InfoModal() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button className="group">
          <Info className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors" />
        </button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[550px] max-h-[85vh] overflow-y-auto">
        <DialogHeader className="pb-6 border-b border-border/10">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-nvidia-green/15 flex items-center justify-center">
              <Sparkles className="h-5 w-5 text-nvidia-green" />
            </div>
            <DialogTitle className="text-2xl font-bold text-foreground nvidia-build-gradient-text">
              Text to Knowledge Graph
            </DialogTitle>
          </div>
          <DialogDescription className="text-base text-muted-foreground leading-relaxed">
            An AI-powered platform that transforms your documents into structured knowledge graphs. 
            Extract meaningful relationships from text using state-of-the-art language models and visualize 
            your data in interactive, explorable formats.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-8 pt-6">
          {/* Key Features Section */}
          <div className="nvidia-build-card">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-8 h-8 rounded-lg bg-nvidia-green/15 flex items-center justify-center">
                <Sparkles className="h-4 w-4 text-nvidia-green" />
              </div>
              <h4 className="text-lg font-semibold text-foreground">Key Features</h4>
            </div>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-nvidia-green mt-2 flex-shrink-0"></div>
                <p className="text-sm text-foreground leading-relaxed">
                  <span className="font-semibold">Knowledge Triple Extraction:</span> Automatically identify subject-predicate-object relationships from your text documents
                </p>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-nvidia-green mt-2 flex-shrink-0"></div>
                <p className="text-sm text-foreground leading-relaxed">
                  <span className="font-semibold">Interactive Visualization:</span> Explore relationships through dynamic, interactive knowledge graphs
                </p>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-nvidia-green mt-2 flex-shrink-0"></div>
                <p className="text-sm text-foreground leading-relaxed">
                  <span className="font-semibold">Multi-Format Export:</span> Export your knowledge graphs in JSON, CSV, and PNG formats
                </p>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 rounded-full bg-nvidia-green mt-2 flex-shrink-0"></div>
                <p className="text-sm text-foreground leading-relaxed">
                  <span className="font-semibold">AI-Powered:</span> Leverage cutting-edge language models including NVIDIA, OpenAI, and Ollama
                </p>
              </div>
            </div>
          </div>

          {/* How to Use Section */}
          <div className="nvidia-build-card">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-8 h-8 rounded-lg bg-nvidia-green/15 flex items-center justify-center">
                <Info className="h-4 w-4 text-nvidia-green" />
              </div>
              <h4 className="text-lg font-semibold text-foreground">How to Use</h4>
            </div>
            <div className="space-y-4">
              <div className="flex items-start gap-4">
                <div className="w-6 h-6 rounded-full bg-nvidia-green/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Upload className="h-3 w-3 text-nvidia-green" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">1. Upload Documents</p>
                  <p className="text-xs text-muted-foreground">Upload markdown, CSV, text, or JSON files to get started</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="w-6 h-6 rounded-full bg-nvidia-green/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Sparkles className="h-3 w-3 text-nvidia-green" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">2. Configure Models</p>
                  <p className="text-xs text-muted-foreground">Select your preferred language model and configure processing options</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="w-6 h-6 rounded-full bg-nvidia-green/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Zap className="h-3 w-3 text-nvidia-green" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">3. Extract Knowledge</p>
                  <p className="text-xs text-muted-foreground">Process your documents to generate structured knowledge triples</p>
                </div>
              </div>
              <div className="flex items-start gap-4">
                <div className="w-6 h-6 rounded-full bg-nvidia-green/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Eye className="h-3 w-3 text-nvidia-green" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">4. Visualize & Explore</p>
                  <p className="text-xs text-muted-foreground">Navigate your knowledge graph in 2D or 3D interactive visualizations</p>
                </div>
              </div>
            </div>
          </div>

          {/* Powered by NVIDIA Section */}
          <div className="bg-nvidia-green/5 border border-nvidia-green/20 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Built with NVIDIA's advanced AI infrastructure and optimized for enterprise-grade knowledge extraction workflows.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
} 