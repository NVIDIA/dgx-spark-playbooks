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
import { SetStateAction, useState } from 'react';
import styles from '@/styles/DocumentIngestion.module.css';

declare module 'react' {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    webkitdirectory?: string;
    directory?: string;
  }
}

interface DocumentIngestionProps {
  files: FileList | null;
  ingestMessage: string;
  isIngesting: boolean;
  setFiles: (files: FileList | null) => void;
  setIngestMessage: (message: string) => void;
  setIsIngesting: (value: boolean) => void;
  onSuccessfulIngestion?: () => void;
}

export default function DocumentIngestion({
  files,
  ingestMessage,
  isIngesting,
  setFiles,
  setIngestMessage,
  setIsIngesting,
  onSuccessfulIngestion
}: DocumentIngestionProps) {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFiles(e.target.files);
  };

  const handleIngestSubmit = async (e: { preventDefault: () => void }) => {
    e.preventDefault();
    setIsIngesting(true);
    setIngestMessage("");
    
    try {
      if (files && files.length > 0) {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
          formData.append("files", files[i]);
        }
        
        const res = await fetch("/api/ingest", {
          method: "POST",
          body: formData,
        });
        
        const data = await res.json();
        setIngestMessage(data.message);
        
        if (res.ok && onSuccessfulIngestion) {
          onSuccessfulIngestion();
        }
      } else {
        setIngestMessage("Please select files or specify a directory path.");
      }
    } catch (error) {
      console.error("Error during ingestion:", error);
      setIngestMessage("Error during ingestion. Please check the console for details.");
    } finally {
      setIsIngesting(false);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFiles(e.dataTransfer.files);
    }
  };

  return (
    <div className={styles.section}>
      <h1>Document Ingestion</h1>
      <form onSubmit={handleIngestSubmit} className={styles.ingestForm}>
        <div
          className={styles.uploadSection}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <label htmlFor="file-upload" className={styles.customFileLabel}>
            Choose Files
          </label>
          <input
            id="file-upload"
            type="file"
            multiple
            onChange={handleFileChange}
            disabled={isIngesting}
            className={styles.fileInput}
          />
          <span className={styles.fileName}>
            {files && files.length > 0 ? Array.from(files).map(f => f.name).join(', ') : "No file chosen"}
          </span>
          <p className={styles.helpText}>
            Select files or drag and drop them here
          </p>
        </div>
        
        <button 
          type="submit" 
          disabled={isIngesting || !files}
          className={styles.ingestButton}
        >
          {isIngesting ? "Ingesting..." : "Ingest Documents"}
        </button>
      </form>

      {ingestMessage && (
        <div className={styles.messageContainer}>
          <p>{ingestMessage}</p>
        </div>
      )}
    </div>
  );
} 