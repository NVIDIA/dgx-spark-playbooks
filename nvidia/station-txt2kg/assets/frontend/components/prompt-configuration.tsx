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
import { useState, useEffect } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AlertCircle, Save, Undo } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

// Default prompts used for triple extraction
const DEFAULT_EXTRACTION_PROMPT = `You are a knowledge graph builder that extracts structured information from text.
Extract subject-predicate-object triples from the following text.

Guidelines:
- Extract only factual triples present in the text
- Normalize entity names to their canonical form
- Assign appropriate confidence scores (0-1)
- Include entity types in metadata
- For each triple, include a brief context from the source text

Text: {text}

{format_instructions}`;

const DEFAULT_SYSTEM_PROMPT = `You are an expert that can extract knowledge triples with the form \`('entity', 'relation', 'entity)\` from a text, mainly using entities from the entity list given by the user. Keep relations 2 words max.
Separate each with a new line. Do not output anything else (no notes, no explanations, etc).`;

const ALTERNATIVE_SYSTEM_PROMPT = `Please convert the above text into a list of knowledge triples with the form ('entity', 'relation', 'entity'). Seperate each with a new line. Do not output anything else. Try to focus on key triples that form a connected graph.`;

const DEFAULT_GRAPH_TRANSFORMER_PROMPT = `You are tasked with converting text into a structured graph format.

Extract entities and their relationships from the text and structure them following these guidelines:
- Identify key entities (people, organizations, concepts, locations, etc.)
- Extract relationships between entities
- Normalize entity names to a canonical form
- Use descriptive relationship types in UPPERCASE (e.g., WORKS_AT, PART_OF)
- Add relevant properties where applicable

Organize your response as graph elements with nodes and relationships.`;

export interface PromptConfigurations {
  defaultExtractionPrompt: string;
  systemPrompt: string;
  graphTransformerPrompt: string;
}

interface PromptConfigurationProps {
  onChange?: (configs: PromptConfigurations) => void;
  initialConfigs?: PromptConfigurations;
  langChainMethod?: 'default' | 'graphtransformer';
  useLangChain?: boolean;
}

export function PromptConfiguration({ 
  onChange, 
  initialConfigs, 
  langChainMethod = 'default',
  useLangChain = true
}: PromptConfigurationProps) {
  const [extractionPrompt, setExtractionPrompt] = useState(initialConfigs?.defaultExtractionPrompt || DEFAULT_EXTRACTION_PROMPT);
  const [systemPrompt, setSystemPrompt] = useState(initialConfigs?.systemPrompt || DEFAULT_SYSTEM_PROMPT);
  const [graphTransformerPrompt, setGraphTransformerPrompt] = useState(initialConfigs?.graphTransformerPrompt || DEFAULT_GRAPH_TRANSFORMER_PROMPT);
  const [activeTab, setActiveTab] = useState(useLangChain && langChainMethod === 'default' ? "default" : "system");
  const [hasChanges, setHasChanges] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [systemPromptTemplate, setSystemPromptTemplate] = useState<'default' | 'alternative'>(
    systemPrompt === ALTERNATIVE_SYSTEM_PROMPT ? 'alternative' : 'default'
  );

  // Update state when initialConfigs changes
  useEffect(() => {
    if (initialConfigs) {
      setExtractionPrompt(initialConfigs.defaultExtractionPrompt || DEFAULT_EXTRACTION_PROMPT);
      setSystemPrompt(initialConfigs.systemPrompt || DEFAULT_SYSTEM_PROMPT);
      setGraphTransformerPrompt(initialConfigs.graphTransformerPrompt || DEFAULT_GRAPH_TRANSFORMER_PROMPT);
      
      // Set the proper template selection based on the loaded system prompt
      if (initialConfigs.systemPrompt === ALTERNATIVE_SYSTEM_PROMPT) {
        setSystemPromptTemplate('alternative');
      } else {
        setSystemPromptTemplate('default');
      }
    }
  }, [initialConfigs]);

  // Update active tab when langChainMethod or useLangChain changes
  useEffect(() => {
    if (useLangChain) {
      setActiveTab(langChainMethod === 'default' ? "default" : "graph");
    } else {
      setActiveTab("system");
    }
  }, [langChainMethod, useLangChain]);

  // Check for changes
  useEffect(() => {
    const originalConfigs = initialConfigs || {
      defaultExtractionPrompt: DEFAULT_EXTRACTION_PROMPT,
      systemPrompt: DEFAULT_SYSTEM_PROMPT,
      graphTransformerPrompt: DEFAULT_GRAPH_TRANSFORMER_PROMPT
    };

    const hasChanged = extractionPrompt !== originalConfigs.defaultExtractionPrompt ||
                       systemPrompt !== originalConfigs.systemPrompt ||
                       graphTransformerPrompt !== originalConfigs.graphTransformerPrompt;
                       
    setHasChanges(hasChanged);
  }, [extractionPrompt, systemPrompt, graphTransformerPrompt, initialConfigs]);

  // Save changes
  const handleSave = () => {
    try {
      const configs: PromptConfigurations = {
        defaultExtractionPrompt: extractionPrompt,
        systemPrompt: systemPrompt,
        graphTransformerPrompt: graphTransformerPrompt
      };

      // Save to local storage
      localStorage.setItem("promptConfigurations", JSON.stringify(configs));
      
      // Trigger onChange callback if provided
      if (onChange) {
        onChange(configs);
      }
      
      setError(null);
      setHasChanges(false);
    } catch (err) {
      setError("Failed to save prompt configurations");
      console.error("Error saving prompt configurations:", err);
    }
  };

  // Reset to defaults
  const handleReset = () => {
    setExtractionPrompt(DEFAULT_EXTRACTION_PROMPT);
    setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
    setGraphTransformerPrompt(DEFAULT_GRAPH_TRANSFORMER_PROMPT);
  };

  // Handle system prompt template change
  const handleSystemPromptTemplateChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const template = e.target.value as 'default' | 'alternative';
    setSystemPromptTemplate(template);
    
    // Set the corresponding prompt text
    if (template === 'default') {
      setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
    } else {
      setSystemPrompt(ALTERNATIVE_SYSTEM_PROMPT);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium">Prompt Configurations</h3>
        <div className="flex items-center gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleReset}
            disabled={!hasChanges}
            className="h-8 px-2 text-xs"
          >
            <Undo className="h-3.5 w-3.5 mr-1" />
            Reset to Defaults
          </Button>
          <Button 
            size="sm" 
            onClick={handleSave}
            disabled={!hasChanges}
            className="h-8 px-2 text-xs"
          >
            <Save className="h-3.5 w-3.5 mr-1" />
            Save Changes
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="my-2">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full" style={{ 
          gridTemplateColumns: 
            useLangChain 
              ? (langChainMethod === 'default' ? '1fr 1fr' : '1fr 1fr') 
              : '1fr'
        }}>
          {useLangChain && langChainMethod === 'default' && (
            <TabsTrigger value="default">Default Extraction</TabsTrigger>
          )}
          <TabsTrigger value="system">System Prompt</TabsTrigger>
          {useLangChain && langChainMethod === 'graphtransformer' && (
            <TabsTrigger value="graph">Graph Transformer</TabsTrigger>
          )}
        </TabsList>
        
        {useLangChain && langChainMethod === 'default' && (
          <TabsContent value="default" className="space-y-2 pt-2">
            <div className="text-xs text-muted-foreground mb-2">
              This prompt is used with LangChain for structured extraction of triples.
              <span className="block mt-1">
                Variables: <code className="bg-muted px-1 py-0.5 rounded">{"{text}"}</code> and <code className="bg-muted px-1 py-0.5 rounded">{"{format_instructions}"}</code>
              </span>
            </div>
            <Textarea
              value={extractionPrompt}
              onChange={(e) => setExtractionPrompt(e.target.value)}
              className="min-h-[300px] font-mono text-xs"
              placeholder="Enter extraction prompt template..."
            />
          </TabsContent>
        )}
        
        <TabsContent value="system" className="space-y-2 pt-2">
          <div className="mb-4">
            <Label htmlFor="system-prompt-template" className="text-xs font-medium mb-1 block">Prompt Template</Label>
            <select
              id="system-prompt-template"
              value={systemPromptTemplate}
              onChange={handleSystemPromptTemplateChange}
              className="w-full p-2 text-sm rounded-md border border-input bg-background"
            >
              <option value="default">Detailed Triple Extraction</option>
              <option value="alternative">Connected Graph Focus</option>
            </select>
            <p className="text-xs text-muted-foreground mt-1">Select a template or customize the prompt below</p>
          </div>
          
          <Textarea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            className="min-h-[300px] font-mono text-xs"
            placeholder="Enter system prompt..."
          />
        </TabsContent>
        
        {useLangChain && langChainMethod === 'graphtransformer' && (
          <TabsContent value="graph" className="space-y-2 pt-2">
            <div className="text-xs text-muted-foreground mb-2">
              This prompt is used with the LLMGraphTransformer for structured graph extraction.
            </div>
            <Textarea
              value={graphTransformerPrompt}
              onChange={(e) => setGraphTransformerPrompt(e.target.value)}
              className="min-h-[300px] font-mono text-xs"
              placeholder="Enter graph transformer prompt..."
            />
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
} 