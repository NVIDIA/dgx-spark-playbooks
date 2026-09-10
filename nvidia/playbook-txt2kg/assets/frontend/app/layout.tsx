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
import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { DocumentProvider } from "@/contexts/document-context"
import { ClientInitializer } from "@/components/client-init"
import Link from "next/link"
import { Search as SearchIcon } from "lucide-react"
import { NvidiaIcon } from "@/components/nvidia-icon"
import { ThemeToggle } from "@/components/theme-toggle"
import { InfoModal } from "@/components/info-modal"
import { SettingsModal } from "@/components/settings-modal"
import { Toaster } from "@/components/ui/toaster"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
})

export const metadata: Metadata = {
  title: "txt2kg | NVIDIA Knowledge Graph Builder",
  description: "Convert text documents to knowledge graphs using NVIDIA AI",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={inter.variable}>
      <body className={inter.className}>
        <ThemeProvider defaultTheme="dark">
          <DocumentProvider>
            <ClientInitializer />
            {/* Modern Gradient Header */}
            <header className="border-b border-border/50 backdrop-blur-md dark:bg-background/95 bg-background sticky top-0 z-50 shadow-sm">
              <div className="container mx-auto px-4 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <NvidiaIcon className="h-8 w-8" />
                  <div>
                    <span className="text-xl font-bold gradient-text">txt2kg</span>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <Link
                    href="/rag"
                    className="flex items-center gap-2 text-sm font-medium rounded-lg px-3 py-2 transition-colors border border-[#76b900]/40 text-[#76b900] bg-[#76b900]/10 hover:bg-[#76b900]/20 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#76b900]/50 dark:bg-[#76b900]/20 dark:hover:bg-[#76b900]/30 dark:border-[#76b900]/50"
                  >
                    <SearchIcon className="h-4 w-4 text-current" />
                    <span>RAG Search</span>
                  </Link>
                  <InfoModal />
                  <SettingsModal />
                  <ThemeToggle />
                </div>
              </div>
            </header>
            {children}
            <Toaster />
          </DocumentProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}