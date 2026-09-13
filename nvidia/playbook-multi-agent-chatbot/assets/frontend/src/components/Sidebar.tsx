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
import React, { useState, useEffect, useRef, useCallback } from 'react';
import styles from '@/styles/Sidebar.module.css';

interface Model {
  id: string;
  name: string;
}

interface ChatMetadata {
  name: string;
}

interface SidebarProps {
  showIngestion: boolean;
  setShowIngestion: (value: boolean) => void;
  refreshTrigger?: number;
  currentChatId: string | null;
  onChatChange: (chatId: string) => Promise<void>;
}


export default function Sidebar({ 
  showIngestion, 
  setShowIngestion, 
  refreshTrigger = 0,
  currentChatId,
  onChatChange
}: SidebarProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(["config", "history"]));
  const [isLoading, setIsLoading] = useState(false);
  const [availableSources, setAvailableSources] = useState<string[]>([]);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isLoadingSources, setIsLoadingSources] = useState(false);
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [chats, setChats] = useState<string[]>([]);
  const [isLoadingChats, setIsLoadingChats] = useState(false);
  const [chatMetadata, setChatMetadata] = useState<Record<string, ChatMetadata>>({});
  
  // Add ref for chat list
  const chatListRef = useRef<HTMLDivElement>(null);

  // Load initial configuration
  useEffect(() => {
    const loadInitialConfig = async () => {
      try {
        setIsLoading(true);
        
        // Get selected model
        const modelResponse = await fetch("/api/selected_model");
        if (modelResponse.ok) {
          const { model } = await modelResponse.json();
          setSelectedModel(model);
        }

        // Get selected sources
        const sourcesResponse = await fetch("/api/selected_sources");
        if (sourcesResponse.ok) {
          const { sources } = await sourcesResponse.json();
          setSelectedSources(sources);
        }


        // Get available models
        await fetchAvailableModels();

        // Get sources
        await fetchSources();
        
        // Get chats if history section is expanded (which it is by default)
        if (expandedSections.has('history')) {
          await fetchChats();
        }
      } catch (error) {
        console.error("Error loading initial config:", error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadInitialConfig();
  }, []);

  // Fetch available models
  const fetchAvailableModels = async () => {
    try {
      setIsLoadingModels(true);
      const response = await fetch("/api/available_models");
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Error fetching available models: ${response.status} - ${errorText}`);
        return;
      }
      
      const data = await response.json();
      const models = data.models.map((modelId: string) => ({
        id: modelId,
        name: modelId.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
      }));
      setAvailableModels(models);
    } catch (error) {
      console.error("Error fetching available models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  // Fetch available sources
  const fetchSources = useCallback(async () => {
    try {
      setIsLoadingSources(true);
      console.log("Fetching sources...");
      const response = await fetch("/api/sources");
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Error fetching sources: ${response.status} - ${errorText}`);
        setAvailableSources([]);
        return;
      }
      
      const data = await response.json();
      console.log("Sources fetched:", data.sources);
      setAvailableSources(data.sources || []);
    } catch (error) {
      console.error("Error fetching sources:", error);
      setAvailableSources([]);
    } finally {
      setIsLoadingSources(false);
    }
  }, []);

  // Get sources on initial load and when the context section is expanded
  useEffect(() => {
    if (expandedSections.has('context')) {
      fetchSources();
    }
  }, [expandedSections.has('context'), fetchSources]);

  // Refresh sources when refreshTrigger changes (document ingestion)
  useEffect(() => {
    if (refreshTrigger > 0) { // Only refresh if not the initial render
      fetchSources();
    }
  }, [refreshTrigger, fetchSources]);

  // Add function to fetch chat metadata
  const fetchChatMetadata = useCallback(async (chatId: string) => {
    try {
      const response = await fetch(`/api/chat/${chatId}/metadata`);
      if (response.ok) {
        const metadata = await response.json();
        setChatMetadata(prev => ({
          ...prev,
          [chatId]: metadata
        }));
      }
    } catch (error) {
      console.error(`Error fetching metadata for chat ${chatId}:`, error);
    }
  }, []);

  // Update fetchChats to also fetch metadata
  const fetchChats = useCallback(async () => {
    try {
      console.log("fetchChats: Starting to fetch chats...");
      setIsLoadingChats(true);
      const response = await fetch("/api/chats");
      if (response.ok) {
        const data = await response.json();
        console.log("fetchChats: Received chats:", data.chats);
        setChats(data.chats);
        // Fetch metadata for each chat
        await Promise.all(data.chats.map(fetchChatMetadata));
        console.log("fetchChats: Completed fetching all chat metadata");
      } else {
        console.error("fetchChats: Failed to fetch chats, status:", response.status);
      }
    } catch (error) {
      console.error("Error fetching chats:", error);
    } finally {
      setIsLoadingChats(false);
    }
  }, [fetchChatMetadata]);

  // Fetch chats when history section is expanded
  useEffect(() => {
    if (expandedSections.has('history')) {
      fetchChats();
    }
  }, [expandedSections.has('history'), fetchChats]);


  // Update highlight position when currentChatId changes
  useEffect(() => {
    if (currentChatId && chatListRef.current) {
      const activeChat = chatListRef.current.querySelector(`.${styles.active}`) as HTMLElement;
      if (activeChat) {
        const offset = activeChat.offsetTop;
        chatListRef.current.style.setProperty('--highlight-offset', `${offset}px`);
      }
    }
  }, [currentChatId, chats]);

  // Add new effect to handle initial position and chat list loading
  useEffect(() => {
    if (chatListRef.current && currentChatId && chats.length > 0) {
      // Small delay to ensure DOM is ready
      setTimeout(() => {
        const activeChat = chatListRef.current?.querySelector(`.${styles.active}`) as HTMLElement;
        if (activeChat) {
          const offset = activeChat.offsetTop;
          chatListRef.current?.style.setProperty('--highlight-offset', `${offset}px`);
        }
      }, 50);
    }
  }, [currentChatId, chats.length]);

  const handleClose = () => {
    setIsClosing(true);
    setTimeout(() => {
      setIsVisible(false);
      setIsClosing(false);
    }, 500); // Match the new animation duration
  };

  const toggleSidebar = () => {
    if (!isVisible) {
      setIsVisible(true);
      setIsClosing(false);
      fetchSources();
      // Also fetch chats when sidebar opens
      if (expandedSections.has('history')) {
        fetchChats();
      }
    } else {
      handleClose();
    }
  };

  const toggleSection = (section: string) => {
    const newExpandedSections = new Set(expandedSections);
    if (newExpandedSections.has(section)) {
      newExpandedSections.delete(section);
    } else {
      newExpandedSections.add(section);
      // Get sources when context section is expanded
      if (section === 'context') {
        fetchSources();
      }
    }
    setExpandedSections(newExpandedSections);
  };

  const isSectionExpanded = (section: string) => {
    return expandedSections.has(section);
  };

  const handleSourceToggle = async (source: string) => {
    let newSelectedSources: string[];
    
    if (selectedSources.includes(source)) {
      // Remove source if already selected
      newSelectedSources = selectedSources.filter(s => s !== source);
    } else {
      // Add source if not selected
      newSelectedSources = [...selectedSources, source];
    }
    
    setSelectedSources(newSelectedSources);
    
    try {
      const response = await fetch("/api/selected_sources", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newSelectedSources)
      });
      
      if (!response.ok) {
        console.error("Failed to update selected sources");
        // Revert the local state if the update failed
        setSelectedSources(selectedSources);
      }
    } catch (error) {
      console.error("Error updating selected sources:", error);
      // Revert the local state if the update failed
      setSelectedSources(selectedSources);
    }
  };


  const handleChatSelect = async (chatId: string) => {
    try {
      await onChatChange(chatId);
      
      // Close sidebar on mobile after selection
      if (window.innerWidth < 768) {
        handleClose();
      }
    } catch (error) {
      console.error("Error selecting chat:", error);
    }
  };

  const handleRenameChat = async (chatId: string, currentName: string) => {
    const newName = prompt("Enter new chat name:", currentName);
    if (newName && newName.trim() && newName !== currentName) {
      try {
        const response = await fetch("/api/chat/rename", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ chat_id: chatId, new_name: newName.trim() })
        });

        if (!response.ok) {
          console.error("Failed to rename chat");
          return;
        }

        // Fetch updated metadata for the renamed chat
        await fetchChatMetadata(chatId);
      } catch (error) {
        console.error("Error renaming chat:", error);
      }
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    try {
      // Delete the chat
      const response = await fetch(`/api/chat/${chatId}`, {
        method: "DELETE"
      });

      if (!response.ok) {
        console.error("Failed to delete chat");
        return;
      }

      // Refresh chat list
      await fetchChats();
      
      // If we deleted the current chat
      if (currentChatId === chatId) {
        // Get updated list of chats
        const chatsResponse = await fetch("/api/chats");
        const { chats: remainingChats } = await chatsResponse.json();

        if (remainingChats.length > 0) {
          // Switch to another chat
          await onChatChange(remainingChats[0]);
        } else {
          // No chats left, create a new one
          await handleNewChat();
        }
      }
    } catch (error) {
      console.error("Error deleting chat:", error);
    }
  };

  const handleNewChat = async () => {
    try {
      console.log("handleNewChat: Starting new chat creation...");
      
      // Create new chat using backend endpoint
      const response = await fetch("/api/chat/new", {
        method: "POST"
      });
      
      if (!response.ok) {
        console.error("Failed to create new chat");
        return;
      }

      const data = await response.json();
      console.log("handleNewChat: Created new chat:", data.chat_id);
      
      // First, refresh the chat list to ensure the new chat is available
      console.log("handleNewChat: Refreshing chat list...");
      await fetchChats();
      console.log("handleNewChat: Chat list refreshed");
      
      // Then change to the new chat
      await onChatChange(data.chat_id);
      console.log("handleNewChat: Changed to new chat");
      
      // Close sidebar on mobile
      if (window.innerWidth < 768) {
        handleClose();
      }
      
      // Add a small delay to ensure the DOM has updated, then trigger highlight animation
      setTimeout(() => {
        if (chatListRef.current) {
          const activeChat = chatListRef.current.querySelector(`.${styles.active}`) as HTMLElement;
          if (activeChat) {
            const offset = activeChat.offsetTop;
            chatListRef.current.style.setProperty('--highlight-offset', `${offset}px`);
          }
        }
      }, 100); // Increased delay for more reliability
    } catch (error) {
      console.error("Error creating new chat:", error);
    }
  };

  const handleClearAllChats = async () => {
    // Show confirmation dialog
    const confirmClear = window.confirm(
      `Are you sure you want to clear all ${chats.length} chat conversations? This action cannot be undone.`
    );
    
    if (!confirmClear) {
      return;
    }

    try {
      const response = await fetch("/api/chats/clear", {
        method: "DELETE"
      });

      if (!response.ok) {
        console.error("Failed to clear all chats");
        alert("Failed to clear chats. Please try again.");
        return;
      }

      const data = await response.json();
      
      // Switch to the new chat created by the backend
      await onChatChange(data.new_chat_id);
      
      // Refresh chat list
      await fetchChats();
      
      // Close sidebar on mobile
      if (window.innerWidth < 768) {
        handleClose();
      }

      console.log(`Successfully cleared ${data.cleared_count} chats`);
    } catch (error) {
      console.error("Error clearing all chats:", error);
      alert("An error occurred while clearing chats. Please try again.");
    }
  };

  const handleModelChange = async (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = event.target.value;
    const newModelLower = newModel.toLowerCase();
    setSelectedModel(newModel);
    
    try {
      console.log("Updating selected model to:", newModel);
      
      const response = await fetch("/api/selected_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: newModel })
      });
      
      if (!response.ok) {
        console.error("Failed to update selected model");
        // Revert the local state if the update failed
        setSelectedModel(selectedModel);
      }
    } catch (error) {
      console.error("Error updating selected model:", error);
      // Revert the local state if the update failed
      setSelectedModel(selectedModel);
    }
  };

  return (
    <>
      <button 
        className={`${styles.toggleSidebarButton} ${isVisible && !isClosing ? styles.active : ''}`} 
        onClick={toggleSidebar}
      >
        ☰
      </button>
      
      {isVisible && (
        <>
          <div 
            className={`${styles.sidebarOverlay} ${isClosing ? styles.closing : ''}`} 
            onClick={handleClose} 
          />
          <div className={`${styles.sidebar} ${isClosing ? styles.closing : ''}`}>
            <button 
              className={styles.closeSidebarButton} 
              onClick={handleClose}
            >
              ×
            </button>
            <div className={styles.sidebarHeader}>
              <h2 className={styles.title}>Spark Chat</h2>
            </div>
            
            {/* Model Selection */}
            <div className={styles.sidebarSection}>
              <div 
                className={styles.sectionHeader} 
                onClick={() => toggleSection('model')}
              >
                <h3>Model</h3>
                <span className={isSectionExpanded('model') ? styles.arrowUp : styles.arrowDown}>▼</span>
              </div>
              <div className={`${styles.sectionContent} ${isSectionExpanded('model') ? styles.expanded : ''}`}>
                <div className={styles.configItem}>
                  <label htmlFor="model-select">Select Supervisor Model</label>
                  <select
                    id="model-select"
                    className={styles.modelSelect}
                    value={selectedModel}
                    onChange={handleModelChange}
                    disabled={isLoadingModels}
                  >
                    {isLoadingModels ? (
                      <option value="">Loading models...</option>
                    ) : (
                      availableModels.map(model => (
                        <option key={model.id} value={model.id}>
                          {model.name}
                        </option>
                      ))
                    )}
                  </select>
                </div>
              </div>
            </div>
            
            {/* Context */}
            <div className={styles.sidebarSection}>
              <div 
                className={styles.sectionHeader} 
                onClick={() => toggleSection('context')}
              >
                <h3>Context</h3>
                <span className={isSectionExpanded('context') ? styles.arrowUp : styles.arrowDown}>▼</span>
              </div>
              <div className={`${styles.sectionContent} ${isSectionExpanded('context') ? styles.expanded : ''}`}>
                <div className={styles.configItem}>
                  <label>Select Sources</label>
                  <div className={styles.sourcesContainer}>
                    {availableSources.length === 0 ? (
                      <div className={styles.noSources}>No sources available</div>
                    ) : (
                      availableSources.map(source => (
                        <div key={source} className={styles.sourceItem}>
                          <input
                            type="checkbox"
                            id={`source-${source}`}
                            checked={selectedSources.includes(source)}
                            onChange={() => handleSourceToggle(source)}
                          />
                          <label htmlFor={`source-${source}`}>{source}</label>
                        </div>
                      ))
                    )}
                  </div>
                  <div className={styles.buttonGroup}>
                    <button 
                      className={styles.uploadDocumentsButton}
                      onClick={() => setShowIngestion(true)}
                      title="Upload Documents"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        width="16"
                        height="16"
                      >
                        <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                      </svg>
                      Upload Documents
                    </button>
                    <button 
                      className={styles.refreshButton}
                      onClick={(e) => {
                        e.preventDefault();
                        fetchSources();
                      }}
                      disabled={isLoadingSources}
                    >
                      {isLoadingSources ? "Loading..." : "Refresh Sources"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Chat History */}
            <div className={styles.sidebarSection}>
              <div 
                className={styles.sectionHeader} 
                onClick={() => toggleSection('history')}
              >
                <h3>Chat History</h3>
                <span className={isSectionExpanded('history') ? styles.arrowUp : styles.arrowDown}>▼</span>
              </div>
              <div className={`${styles.sectionContent} ${isSectionExpanded('history') ? styles.expanded : ''}`}>
                <div className={styles.chatList} ref={chatListRef}>
                  {isLoadingChats ? (
                    <div className={styles.loadingText}>Loading chats...</div>
                  ) : chats.length === 0 ? (
                    <div className={styles.noChatText}>No previous chats</div>
                  ) : (
                    chats.map((chatId) => (
                      <div
                        key={chatId}
                        className={`${styles.chatItem} ${currentChatId === chatId ? styles.active : ''}`}
                        onClick={() => handleChatSelect(chatId)}
                      >
                        <div className={styles.chatName}>
                          {chatMetadata[chatId]?.name || chatId.slice(0, 8)}
                        </div>
                        <div className={styles.chatActions}>
                          <button
                            className={styles.chatActionButton}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRenameChat(chatId, chatMetadata[chatId]?.name || `Chat ${chatId.slice(0, 8)}`);
                            }}
                          >
                            ✎
                          </button>
                          <button
                            className={styles.chatActionButton}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteChat(chatId);
                            }}
                          >
                            ×
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Chat Action Buttons at bottom */}
            <div className={styles.chatButtonsContainer}>
              <button 
                className={styles.newChatButton}
                onClick={handleNewChat}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
                New Chat
              </button>
              
              <button 
                className={styles.clearChatsButton}
                onClick={handleClearAllChats}
                disabled={chats.length === 0}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M3 6h18" />
                  <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                  <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                </svg>
                Clear All
              </button>
            </div>
          </div>
        </>
      )}
    </>
  );
} 