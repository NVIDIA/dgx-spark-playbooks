/*
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/
"use client";
import { useState, useRef, useEffect } from 'react';
import QuerySection from '@/components/QuerySection';
import DocumentIngestion from '@/components/DocumentIngestion';
import Sidebar from '@/components/Sidebar';
import styles from '@/styles/Home.module.css';

export default function Home() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("[]");
  const [files, setFiles] = useState<FileList | null>(null);
  const [ingestMessage, setIngestMessage] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [showIngestion, setShowIngestion] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Load initial chat ID
  useEffect(() => {
    const fetchCurrentChatId = async () => {
      try {
        const response = await fetch("/api/chat_id");
        if (response.ok) {
          const { chat_id } = await response.json();
          setCurrentChatId(chat_id);
        }
      } catch (error) {
        console.error("Error fetching current chat ID:", error);
      }
    };
    fetchCurrentChatId();
  }, []);

  // Handle chat changes
  const handleChatChange = async (newChatId: string) => {
    try {
      const response = await fetch("/api/chat_id", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: newChatId })
      });
      
      if (response.ok) {
        setCurrentChatId(newChatId);
        setResponse("[]"); // Clear current chat messages with empty JSON array
      }
    } catch (error) {
      console.error("Error updating chat ID:", error);
    }
  };

  // Clean up any ongoing streams when component unmounts
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Function to handle successful document ingestion
  const handleSuccessfulIngestion = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className={styles.container}>
      <Sidebar 
        showIngestion={showIngestion}
        setShowIngestion={setShowIngestion}
        refreshTrigger={refreshTrigger}
        currentChatId={currentChatId}
        onChatChange={handleChatChange}
      />
      
      <div className={styles.mainContent}>
        <QuerySection
          query={query}
          response={response}
          isStreaming={isStreaming}
          setQuery={setQuery}
          setResponse={setResponse}
          setIsStreaming={setIsStreaming}
          abortControllerRef={abortControllerRef}
          setShowIngestion={setShowIngestion}
          currentChatId={currentChatId}
        />
      </div>

      {showIngestion && (
        <>
          <div className={styles.overlay} onClick={() => setShowIngestion(false)} />
          <div className={styles.documentUploadContainer}>
            <button 
              className={styles.closeButton} 
              onClick={() => setShowIngestion(false)}
            >
              ×
            </button>
            <DocumentIngestion
              files={files}
              ingestMessage={ingestMessage}
              isIngesting={isIngesting}
              setFiles={setFiles}
              setIngestMessage={setIngestMessage}
              setIsIngesting={setIsIngesting}
              onSuccessfulIngestion={handleSuccessfulIngestion}
            />
          </div>
        </>
      )}
    </div>
  );
}
