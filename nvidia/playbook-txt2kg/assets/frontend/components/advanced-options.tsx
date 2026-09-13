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
import React, { useState, useRef, useEffect } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface AdvancedOptionsProps {
  title?: string;
  children: React.ReactNode;
  className?: string;
  defaultOpen?: boolean;
}

export function AdvancedOptions({
  title = "Advanced Options",
  children,
  className,
  defaultOpen = false
}: AdvancedOptionsProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const contentRef = useRef<HTMLDivElement>(null);
  const [contentHeight, setContentHeight] = useState<number | undefined>(
    defaultOpen ? undefined : 0
  );

  // Update content height when open state changes
  useEffect(() => {
    if (isOpen) {
      const height = contentRef.current?.scrollHeight;
      setContentHeight(height);
      // After animation completes, set to auto for dynamic content
      const timer = setTimeout(() => setContentHeight(undefined), 200);
      return () => clearTimeout(timer);
    } else {
      // First set to current height, then to 0 for smooth collapse
      setContentHeight(contentRef.current?.scrollHeight);
      requestAnimationFrame(() => setContentHeight(0));
    }
  }, [isOpen]);

  return (
    <div className={cn("border rounded-md overflow-hidden", className)}>
      <button 
        type="button"
        className="w-full flex items-center justify-between p-3 bg-muted/30 cursor-pointer hover:bg-muted/50 transition-colors focus-visible:ring-2 focus-visible:ring-nvidia-green focus-visible:ring-inset"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-controls="advanced-options-content"
      >
        <h3 className="text-sm font-medium flex items-center">
          <ChevronDown 
            className={cn(
              "h-4 w-4 mr-2 transition-transform duration-200",
              !isOpen && "-rotate-90"
            )} 
          />
          {title}
        </h3>
      </button>
      
      <div 
        id="advanced-options-content"
        ref={contentRef}
        className="overflow-hidden transition-all duration-200 ease-out"
        style={{ height: contentHeight !== undefined ? contentHeight : 'auto' }}
        aria-hidden={!isOpen}
      >
        <div className="p-4 border-t border-border/50">
          {children}
        </div>
      </div>
    </div>
  );
} 