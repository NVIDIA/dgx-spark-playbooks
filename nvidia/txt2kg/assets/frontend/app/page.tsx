"use client"

import { useState, useEffect } from "react"
import { ApiKeyPrompt } from "@/components/api-key-prompt"
import { Upload, Zap, Edit, Network } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import React from "react"
import { UploadTab } from "@/components/tabs/UploadTab"
import { ConfigureTab } from "@/components/tabs/ConfigureTab"
import { EditTab } from "@/components/tabs/EditTab"
import { VisualizeTab } from "@/components/tabs/VisualizeTab"

// Add global styles for dropdown visibility
const globalStyles = `
  .model-dropdown {
    position: relative;
    z-index: 9999;
  }
  .model-dropdown-menu {
    z-index: 9999;
  }
`;

export default function Home() {
  // Track active tab for animation
  const [activeTab, setActiveTab] = useState("upload");
  const steps = [
    { value: "upload", label: "Upload", Icon: Upload },
    { value: "configure", label: "Process Documents", Icon: Zap },
    { value: "edit", label: "Edit Knowledge Graph", Icon: Edit },
    { value: "visualize", label: "Visualize Graph", Icon: Network },
  ] as const;
  const activeIndex = Math.max(0, steps.findIndex(s => s.value === activeTab));
  
  // Updated to use callback reference
  const handleTabChange = React.useCallback((tab: string) => {
    const tabElement = document.querySelector(`[data-value="${tab}"]`)
    if (tabElement && 'click' in tabElement) {
      (tabElement as HTMLElement).click()
    }
  }, []);
  
  // Handle tab selection based on URL hash
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      if (['upload', 'configure', 'edit', 'visualize'].includes(hash)) {
        handleTabChange(hash);
        setActiveTab(hash);
      }
    }
    
    // Set hash on initial load if not present
    if (!window.location.hash) {
      window.location.hash = 'upload';
    } else {
      handleHashChange();
    }
    
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [handleTabChange]);
  
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Add style element for global styles */}
      <style dangerouslySetInnerHTML={{ __html: globalStyles }} />
      
      <main className="container mx-auto px-6 py-12 border-b border-border/10">
        
        <Tabs defaultValue="upload" className="w-full mb-12" onValueChange={setActiveTab}>
          <TabsList className="nvidia-build-tabs mb-12" aria-label="Workflow steps">
            {steps.map(({ value, label, Icon }) => (
              <TabsTrigger
                key={value}
                value={value}
                onClick={() => {
                  window.location.hash = value;
                  setActiveTab(value);
                }}
                data-value={value}
                className={`nvidia-build-tab ${activeTab === value ? 'nvidia-build-tab-active' : ''}`}
              >
                <div className="nvidia-build-tab-icon">
                  <Icon className="h-3 w-3 text-nvidia-green" />
                </div>
                <span>{label}</span>
              </TabsTrigger>
            ))}
          </TabsList>
          
          {/* Step 1: Document Upload */}
          <TabsContent value="upload" className="space-y-8">
            <UploadTab onTabChange={handleTabChange} />
          </TabsContent>
          
          {/* Step 2: Configure & Process */}
          <TabsContent value="configure" className="space-y-8">
            <ConfigureTab />
          </TabsContent>
          
          {/* Step 3: Edit Knowledge */}
          <TabsContent value="edit" className="space-y-8">
            <EditTab />
          </TabsContent>
          
          {/* Step 4: Visualize Knowledge Graph */}
          <TabsContent value="visualize" className="space-y-8">
            <VisualizeTab />
          </TabsContent>
        </Tabs>
      </main>

      <ApiKeyPrompt />
    </div>
  )
}

