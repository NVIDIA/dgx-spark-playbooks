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

import { Download, Maximize, Minimize, CuboidIcon, LayoutGrid, Database, Search as SearchIcon, Settings, Zap, HelpCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface GraphToolbarProps {
  // View controls
  use3D: boolean
  onToggle3D: () => void
  isFullscreen: boolean
  onToggleFullscreen: () => void
  
  // Layout controls
  layoutType: "force" | "hierarchical" | "radial"
  onLayoutChange: (layout: "force" | "hierarchical" | "radial") => void
  
  // Data controls
  includeStoredTriples: boolean
  onToggleStoredTriples: (enabled: boolean) => void
  storedTriplesCount: number
  loadingStoredTriples: boolean
  
  // Export
  onExport: (format: "json" | "csv" | "png") => void
  
  // Search
  searchTerm: string
  onSearchChange: (term: string) => void
  onSearch: () => void
  searchInputRef?: React.RefObject<HTMLInputElement | null>
  
  // Stats
  nodeCount: number
  edgeCount: number
}

export function GraphToolbar({
  use3D,
  onToggle3D,
  isFullscreen,
  onToggleFullscreen,
  layoutType,
  onLayoutChange,
  includeStoredTriples,
  onToggleStoredTriples,
  storedTriplesCount,
  loadingStoredTriples,
  onExport,
  searchTerm,
  onSearchChange,
  onSearch,
  searchInputRef,
  nodeCount,
  edgeCount
}: GraphToolbarProps) {
  return (
    <TooltipProvider>
      <div className="bg-background/95 backdrop-blur-sm border border-border/50 rounded-lg p-3 shadow-sm">
        <div className="flex flex-wrap items-center gap-2 md:gap-4">
          
          {/* Primary Actions Group */}
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  size="sm" 
                  variant={use3D ? "default" : "outline"}
                  onClick={onToggle3D}
                  className={`${use3D ? 'bg-nvidia-green hover:bg-nvidia-green/90 text-white border-nvidia-green' : 'border-border hover:bg-muted/50 text-foreground'} px-3 py-2 gap-2`}
                >
                  <CuboidIcon className="h-4 w-4" />
                  <span className="hidden sm:inline">{use3D ? '2D' : '3D'}</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                Switch to {use3D ? '2D' : '3D'} view
              </TooltipContent>
            </Tooltip>
            
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={onToggleFullscreen}
                  className="px-3 py-2 gap-2"
                >
                  {isFullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
                  <span className="hidden sm:inline">{isFullscreen ? "Exit" : "Fullscreen"}</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
              </TooltipContent>
            </Tooltip>
          </div>

          <Separator orientation="vertical" className="h-6 hidden md:block" />

          {/* Layout Controls Group */}
          <div className="flex items-center gap-1">
            <span className="text-xs font-medium text-muted-foreground mr-2 hidden lg:inline">Layout:</span>
            <div className="flex items-center gap-1">
              {[
                { key: "force", label: "Force", icon: null },
                { key: "hierarchical", label: "Tree", icon: null },
                { key: "radial", label: "Radial", icon: null }
              ].map((layout) => (
                <Tooltip key={layout.key}>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant={layoutType === layout.key ? "default" : "ghost"}
                      className={`h-8 px-2 text-xs ${
                        layoutType === layout.key 
                          ? "bg-nvidia-green hover:bg-nvidia-green/90 text-white" 
                          : "hover:bg-muted"
                      }`}
                      onClick={() => onLayoutChange(layout.key as "force" | "hierarchical" | "radial")}
                    >
                      {layout.label}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    {layout.label} layout
                  </TooltipContent>
                </Tooltip>
              ))}
            </div>
          </div>

          <Separator orientation="vertical" className="h-6 hidden md:block" />

          {/* Data Source Controls */}
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-2">
                  <Switch
                    id="stored-triples"
                    checked={includeStoredTriples}
                    onCheckedChange={onToggleStoredTriples}
                    size="sm"
                  />
                  <Label 
                    htmlFor="stored-triples" 
                    className="text-xs font-medium cursor-pointer flex items-center gap-1"
                  >
                    <Database className="h-3 w-3 text-nvidia-green" />
                    <span className="hidden sm:inline">DB ({storedTriplesCount})</span>
                    {loadingStoredTriples && <span className="animate-spin text-nvidia-green">⟳</span>}
                  </Label>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                Include stored triples from database ({storedTriplesCount} available)
              </TooltipContent>
            </Tooltip>
          </div>

          {/* Stats (on larger screens) */}
          <div className="hidden lg:flex items-center gap-2 text-xs text-muted-foreground">
            <Separator orientation="vertical" className="h-6" />
            <span>{nodeCount} nodes</span>
            <span>•</span>
            <span>{edgeCount} edges</span>
          </div>

          {/* Search */}
          <div className="flex items-center gap-2 min-w-0">
            <div className="relative flex-1 min-w-[200px] max-w-[300px]">
              <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                ref={searchInputRef || undefined}
                type="text"
                placeholder="Search nodes... (Ctrl+K)"
                value={searchTerm}
                onChange={(e) => onSearchChange(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    onSearch()
                  }
                }}
                className="pl-10 h-8 text-sm"
              />
            </div>
          </div>

          {/* Secondary Actions - Right Side */}
          <div className="ml-auto flex items-center gap-2">
            {/* Export Dropdown */}
            <DropdownMenu>
              <Tooltip>
                <TooltipTrigger asChild>
                  <DropdownMenuTrigger asChild>
                    <Button 
                      size="sm" 
                      variant="outline"
                      className="px-3 py-2 gap-2"
                    >
                      <Download className="h-4 w-4" />
                      <span className="hidden sm:inline">Export</span>
                    </Button>
                  </DropdownMenuTrigger>
                </TooltipTrigger>
                <TooltipContent>Export graph data</TooltipContent>
              </Tooltip>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => onExport("json")}>
                  <Download className="h-4 w-4 mr-2" />
                  Export as JSON
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => onExport("csv")}>
                  <Download className="h-4 w-4 mr-2" />
                  Export as CSV
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => onExport("png")}>
                  <Download className="h-4 w-4 mr-2" />
                  Export as PNG
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Help/Shortcuts */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  size="sm" 
                  variant="ghost"
                  className="px-2 py-2 h-8"
                >
                  <HelpCircle className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-xs">
                <div className="space-y-1 text-xs">
                  <div className="font-semibold mb-2">Keyboard Shortcuts:</div>
                  <div className="flex justify-between"><span>F</span><span>Fullscreen</span></div>
                  <div className="flex justify-between"><span>3</span><span>Toggle 3D</span></div>
                  <div className="flex justify-between"><span>Ctrl+K</span><span>Search</span></div>
                  <div className="flex justify-between"><span>1</span><span>Force layout</span></div>
                  <div className="flex justify-between"><span>2</span><span>Tree layout</span></div>
                  <div className="flex justify-between"><span>Shift+3</span><span>Radial layout</span></div>
                </div>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}
