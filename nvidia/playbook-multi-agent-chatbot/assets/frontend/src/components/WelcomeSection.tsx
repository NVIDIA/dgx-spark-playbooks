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
import styles from "@/styles/WelcomeSection.module.css";

interface WelcomeSectionProps {
  setQuery: (value: string) => void;
}

export default function WelcomeSection({ setQuery }: WelcomeSectionProps) {
  const promptTemplates = {
    rag: "What is the Blackwell GB202 GPU according the whitepaper document i uploaded?",
    image: "Can you analyze the graphs in this image and tell me any surprising stats? https://menlovc.com/wp-content/uploads/2025/06/5-parents_power_users-062425.png",
    code: `Can you generate code to develop a responsive personal website for my freelance AI dev business based on my personal brand palette?

My palette is:
#606C38
#283618
#FEFAE0
#DDA15E
#BC6C25`,
    chat: "Hey Spark! Can you draft an email asking a product manager in distributed systems to a coffee chat?"
  };

  const handleCardClick = (promptKey: keyof typeof promptTemplates) => {
    setQuery(promptTemplates[promptKey]);
  };

  return (
    <div className={styles.welcomeContainer}>
      <div className={styles.welcomeMessage}>
        Hello! Send a message to start chatting with Spark.
      </div>
      <div className={styles.agentCards}>
        <div 
          className={`${styles.agentCard} ${styles.animate1}`}
          onClick={() => handleCardClick('rag')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <circle cx="11" cy="11" r="8"/>
              <path d="m21 21-4.35-4.35"/>
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Search Documents</h3>
          <p className={styles.agentSubtitle}>RAG Agent</p>
        </div>
        <div 
          className={`${styles.agentCard} ${styles.animate2}`}
          onClick={() => handleCardClick('image')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
              <circle cx="9" cy="9" r="2"/>
              <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Image Processor</h3>
          <p className={styles.agentSubtitle}>Image Understanding Agent</p>
        </div>
        <div 
          className={`${styles.agentCard} ${styles.animate3}`}
          onClick={() => handleCardClick('code')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <polyline points="16,18 22,12 16,6"/>
              <polyline points="8,6 2,12 8,18"/>
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Code Generation</h3>
          <p className={styles.agentSubtitle}>Coding Agent</p>
        </div>
        <div 
          className={`${styles.agentCard} ${styles.animate4}`}
          onClick={() => handleCardClick('chat')}
        >
          <div className={styles.agentIcon}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="24" height="24">
              <path d="m3 21 1.9-5.7a8.5 8.5 0 1 1 3.8 3.8z"/>
            </svg>
          </div>
          <h3 className={styles.agentTitle}>Chat</h3>
          <p className={styles.agentSubtitle}>Local LLM</p>
        </div>
      </div>
    </div>
  );
}
