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
import type React from "react";
import { useRef, useEffect, useState, useCallback } from "react";
import styles from "@/styles/QuerySection.module.css";
import ReactMarkdown from 'react-markdown'; // NEW
import remarkGfm from 'remark-gfm'; // NEW
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"; // NEW
import { oneDark, oneLight, Dark, Light } from "react-syntax-highlighter/dist/esm/styles/prism"; // NEW
import WelcomeSection from "./WelcomeSection";

export function makeChatTheme(isDark: boolean) {
  const base = isDark ? oneDark : oneLight;

  const accents = isDark
    ? {
        tag:        "#E3E3E3",
        prolog:     "#E3E3E3",
        doctype:    "#E3E3E3",
        punctuation:"#99CFCF",
      }
    : {
        tag:        "#9a6700",
        prolog:     "#7a6200",
        doctype:    "#7a6200",
        punctuation:"#6b7280",
      };

  return {
    ...base,

    'pre[class*="language-"]': {
      ...(base['pre[class*="language-"]'] || {}),
      background: "transparent",
    },
    'code[class*="language-"]': {
      ...(base['code[class*="language-"]'] || {}),
      background: "transparent",
    },

    tag:         { ...(base.tag || {}),         color: accents.tag },
    prolog:      { ...(base.prolog || {}),      color: accents.prolog },
    doctype:     { ...(base.doctype || {}),     color: accents.doctype },
    punctuation: { ...(base.punctuation || {}), color: accents.punctuation },

    'attr-name': { ...(base['attr-name'] || {}), color: isDark ? "#e6b450" : "#6b4f00" },
  } as const;
}

function CodeBlockWithCopy({ code, language }: { code: string; language: string }) {
  const [copied, setCopied] = useState(false);
  const [isDark, setIsDark] = useState(false);

  // Listen for theme changes
  useEffect(() => {
    const updateTheme = () => {
      const darkMode = document.documentElement.classList.contains("dark");
      setIsDark(darkMode);
    };

    // Set initial theme
    updateTheme();

    // Listen for changes to the document element's class list
    const observer = new MutationObserver(updateTheme);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, []);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {
      try {
        const textarea = document.createElement("textarea");
        textarea.value = code;
        textarea.style.position = "fixed";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      } catch {}
    }
  };

  return (
    <div className={styles.codeBlock}>
      <button
        type="button"
        className={styles.copyButton}
        onClick={handleCopy}
        aria-label="Copy code"
        title={copied ? "Copied" : "Copy"}
      >
        <svg
          className={styles.copyButtonIcon}
          viewBox="0 0 460 460"
          aria-hidden="true"
          focusable="false"
          fill="currentColor"
        >
          <g>
            <g>
              <g>
                <path d="M425.934,0H171.662c-18.122,0-32.864,14.743-32.864,32.864v77.134h30V32.864c0-1.579,1.285-2.864,2.864-2.864h254.272
                c1.579,0,2.864,1.285,2.864,2.864v254.272c0,1.58-1.285,2.865-2.864,2.865h-74.729v30h74.729
                c18.121,0,32.864-14.743,32.864-32.865V32.864C458.797,14.743,444.055,0,425.934,0z"/>
                <path d="M288.339,139.998H34.068c-18.122,0-32.865,14.743-32.865,32.865v254.272C1.204,445.257,15.946,460,34.068,460h254.272
                c18.122,0,32.865-14.743,32.865-32.864V172.863C321.206,154.741,306.461,139.998,288.339,139.998z M288.341,430H34.068
                c-1.58,0-2.865-1.285-2.865-2.864V172.863c0-1.58,1.285-2.865,2.865-2.865h254.272c1.58,0,2.865,1.285,2.865,2.865v254.273h0.001
                C291.206,428.715,289.92,430,288.341,430z"/>
              </g>
            </g>
          </g>
        </svg>
        <span className={styles.copyButtonLabel}>{copied ? "Copied" : "Copy"}</span>
      </button>
      <SyntaxHighlighter
        language={language}
        style={makeChatTheme(isDark)}
        PreTag="div"
        wrapLongLines
        showLineNumbers={false}
        customStyle={{ margin: "0.6rem 0", borderRadius: 10, background: "transparent" }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}


interface QuerySectionProps {
  query: string;
  response: string;
  isStreaming: boolean;
  setQuery: (value: string) => void;
  setResponse: React.Dispatch<React.SetStateAction<string>>;
  setIsStreaming: (value: boolean) => void;
  abortControllerRef: React.RefObject<AbortController | null>;
  setShowIngestion: (value: boolean) => void;
  currentChatId: string | null;
}

interface Message {
  type: "HumanMessage" | "AssistantMessage" | "ToolMessage";
  content: string;
}



export default function QuerySection({
  query,
  response,
  isStreaming,
  setQuery,
  setResponse,
  setIsStreaming,
  abortControllerRef,
  setShowIngestion,
  currentChatId,
}: QuerySectionProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [showButtons, setShowButtons] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const [toolOutput, setToolOutput] = useState("");
  const [graphStatus, setGraphStatus] = useState("");
  const [isPinnedToolOutputVisible, setPinnedToolOutputVisible] = useState(false);
  const [isToolContentVisible, setIsToolContentVisible] = useState(false);
  const [fadeIn, setFadeIn] = useState(false);
  const firstTokenReceived = useRef(false);
  const hasAssistantContent = useRef(false);
  const fadeTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowButtons(true);
    }, 800);
    return () => clearTimeout(timer);
  }, []);


  useEffect(() => {
    const fetchSelectedSources = async () => {
      try {
        const response = await fetch("/api/selected_sources");
        if (response.ok) {
          const { sources } = await response.json();
          setSelectedSources(sources);
        }
      } catch (error) {
        console.error("Error fetching selected sources:", error);
      }
    };
    fetchSelectedSources();
  }, []);

  useEffect(() => {
    const initWebSocket = async () => {
      if (!currentChatId) return;

      try {
        if (wsRef.current) {
          wsRef.current.close();
        }

        const wsProtocol = 'ws:';
        const wsHost = 'localhost';
        const wsPort = '8000';
        const ws = new WebSocket(`${wsProtocol}//${wsHost}:${wsPort}/ws/chat/${currentChatId}`);
        wsRef.current = ws;

        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          const type = msg.type
          const text = msg.data ?? msg.token ?? "";
        
          switch (type) {
            case "history": {
              console.log('history messages: ', msg.messages);
              if (Array.isArray(msg.messages)) {
                // const filtered = msg.messages.filter(m => m.type !== "ToolMessage"); // TODO: add this back in
                setResponse(JSON.stringify(msg.messages));
                setIsStreaming(false);
              }
              break;
            }
            case "tool_token": {
              if (text !== undefined && text !== "undefined") {
                setToolOutput(prev => prev + text);
              }
              break;
            }
            case "token": {
              if (!text) break;
              if (!firstTokenReceived.current) {
                firstTokenReceived.current = true;
                hasAssistantContent.current = true;
              }
              setResponse(prev => {
                try {
                  const messages = JSON.parse(prev);
                  const last = messages[messages.length - 1];
                  if (last && last.type === "AssistantMessage") {
                    last.content = String(last.content || "") + text;
                  } else {
                    messages.push({ type: "AssistantMessage", content: text });
                  }
                  return JSON.stringify(messages);
                } catch {
                  return String(prev || "") + text;
                }
              });
              break;
            }
            case "node_start": {
              if (msg?.data === "generate") {
                setGraphStatus("Thinking...");
              }
              break;
            } 
            case "tool_start": {
              console.log(type, msg.data);
              setGraphStatus(`calling tool: ${msg?.data}`);
              break;
            }
            case "tool_end":
            case "node_end": {
              console.log(type, msg.data);
              setGraphStatus("");
              break;
            }
            default: {
              // ignore unknown events
            }
          }
        };

        ws.onclose = () => {
          console.log("WebSocket connection closed");
          setIsStreaming(false);
        };

        ws.onerror = (error) => {
          console.error("WebSocket error:", error);
          setIsStreaming(false);
        };
      } catch (error) {
        console.error("Error initializing WebSocket:", error);
        setIsStreaming(false);
      }
    };

    initWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [currentChatId]);

  useEffect(() => {
    try {
      const messages = JSON.parse(response);
      setShowWelcome(messages.length === 0);
    } catch {
      // If response can't be parsed as JSON, check if it's empty
      setShowWelcome(!response.trim());
    }
  }, [response]);

  // Show/hide pinnedToolOutput with fade
  useEffect(() => {
    if (graphStatus) {
      setPinnedToolOutputVisible(true);
      // Trigger fade-in on next tick
      if (fadeTimeoutRef.current) {
        clearTimeout(fadeTimeoutRef.current);
      }
      setFadeIn(false);
      fadeTimeoutRef.current = setTimeout(() => setFadeIn(true), 10);
    } else {
      // Delay hiding to allow fade-out
      setFadeIn(false);
      const timeout = setTimeout(() => {
        setPinnedToolOutputVisible(false);
      }, 800); // match CSS transition duration
      return () => {
        clearTimeout(timeout);
        if (fadeTimeoutRef.current) {
          clearTimeout(fadeTimeoutRef.current);
        }
      };
    }
  }, [graphStatus]);

  const programmaticScroll = useRef(false);
  const scrollTimeout = useRef<number | null>(null);
  const isUserScrollingRef = useRef(false);
  const isNearBottomRef = useRef(true);

  // Check if user is near the bottom of the chat
  const checkScrollPosition = useCallback(() => {
    if (chatContainerRef.current) {
      const container = chatContainerRef.current;
      const threshold = 100; // pixels from bottom
      const isNear = container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
      isNearBottomRef.current = isNear;
    }
  }, []);

  // Handle scroll events to detect user scrolling
  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;

    let scrollTimer: NodeJS.Timeout;

    const handleScroll = () => {
      isUserScrollingRef.current = true;
      checkScrollPosition();
      
      // Reset user scrolling flag after scroll stops
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(() => {
        isUserScrollingRef.current = false;
      }, 150);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    
    return () => {
      container.removeEventListener('scroll', handleScroll);
      clearTimeout(scrollTimer);
    };
  }, [checkScrollPosition]);

  // Auto-scroll to bottom when response changes
  useEffect(() => {
    // Only scroll if we have assistant content and user hasn't manually scrolled away
    if (!hasAssistantContent.current || isUserScrollingRef.current || !isNearBottomRef.current) {
      return;
    }

    const scrollToBottom = () => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ 
          behavior: 'smooth',
          block: 'end'
        });
      }
      
      if (chatContainerRef.current) {
        chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
      }
    };

    scrollToBottom();
  }, [response]);

  const handleQuerySubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const currentQuery = query.trim();
    if (!currentQuery || isStreaming || !wsRef.current) return;

    setQuery("");
    setIsStreaming(true);
    firstTokenReceived.current = false;
    hasAssistantContent.current = false;

    try {
      wsRef.current.send(JSON.stringify({
        message: currentQuery
      }));
 
      setResponse(prev => {
        try {
          const messages = JSON.parse(prev);
          messages.push({
            type: "HumanMessage",
            content: currentQuery
          });
          return JSON.stringify(messages);
        } catch {
          return prev + `\n\nHuman: ${currentQuery}\n\nAssistant: `;
        }
      });
    } catch (error) {
      console.error("Error sending message:", error);
      setIsStreaming(false);
    }
  };

  const handleCancelStream = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsStreaming(false);
    }
  };

  // filter out all ToolMessages
  const parseMessages = (response: string): Message[] => {
    try {
      const parsed = JSON.parse(response);
      if (!Array.isArray(parsed)) return [];
  
      return parsed
        .map((msg: any): Message => ({
          type: msg?.type === "HumanMessage"
            ? "HumanMessage"
            : msg?.type === "ToolMessage"
            ? "ToolMessage"
            : "AssistantMessage",
          content: typeof msg?.content === "string" ? msg.content : String(msg?.content ?? "")
        }))
        .filter((msg) => msg.type !== "ToolMessage"); // discard ToolMessage completely
    } catch {
      if (!response?.trim()) return [];
      
      return [{ type: "AssistantMessage", content: String(response) }];
    }
  };


  return (
    <div className={styles.chatContainer}>
      {showWelcome && <WelcomeSection setQuery={setQuery}/>}

      {/* Minimal fade-in/fade-out: always start hidden, fade in on next tick */}
      {/* {isPinnedToolOutputVisible && ( )}*/}
      <div className={`${styles.pinnedToolOutput} ${!fadeIn ? styles.pinnedToolOutputHidden : ""}`}>
        {graphStatus && (
          <div className={styles.toolHeader} onClick={() => setIsToolContentVisible(v => !v)} style={{ cursor: 'pointer' }}>
            <span className={styles.toolLabel}> {graphStatus} </span>
          </div>
        )}
        </div>
      
      <div className={styles.messagesContainer} ref={chatContainerRef}>
        {parseMessages(response).map((message, index) => {
          const isHuman = message.type === "HumanMessage";
          const key = `${message.type}-${index}`;
          
          if (!message.content?.trim()) return null;
          
          return (
            <div 
              key={key} 
              className={`${styles.messageWrapper} ${isHuman ? styles.userMessageWrapper : styles.assistantMessageWrapper}`}
              style={{
                animationDelay: `${index * 0.1}s`
              }}
            >

            <div className={`${styles.message} ${isHuman ? styles.userMessage : styles.assistantMessage}`}>
              <div className={styles.markdown}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || "");
                      const code = String(children ?? "").replace(/\n$/, "");

                      if (inline || !match) {
                        return (
                          <code className={className} {...props}>
                            {code}
                          </code>
                        );
                      }
                      return (
                        <CodeBlockWithCopy code={code} language={match[1]} />
                      );
                    },
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>
            </div>
          );
        })}

        {isStreaming && (
          <div 
            className={`${styles.messageWrapper} ${styles.assistantMessageWrapper}`}
            style={{
              animationDelay: `${parseMessages(response).length * 0.1}s`
            }}
          >
            <div className={`${styles.message} ${styles.assistantMessage}`}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleQuerySubmit} className={styles.inputContainer}>
        <div className={styles.inputWrapper}>
          <textarea
            rows={1}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Send a message..."
            disabled={isStreaming}
            className={styles.messageInput}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleQuerySubmit(e as any);
              }
            }}
          />
        </div>
        {!isStreaming ? (
          <button 
            type="submit" 
            className={`${styles.sendButton} ${showButtons ? styles.show : ''}`}
            disabled={!query.trim()}
          >
            →
          </button>
        ) : (
          <button 
            type="button" 
            onClick={handleCancelStream} 
            className={`${styles.streamingCancelButton} ${showButtons ? styles.show : ''}`}
          >
            ✕
          </button>
        )}
      </form>
      
      <div className={styles.disclaimer}>
        This is a concept demo to showcase multiple models and MCP use. It is not optimized for performance. Developers can customize and further optimize it for performance.
        <br />
        <span className={styles.info}>Note: If a response is cut short, please start a new chat to continue.</span>
        <br />
        <span className={styles.warning}>Don't forget to shutdown docker containers at the end of the demo.</span>
      </div>
    </div>
  );
}
