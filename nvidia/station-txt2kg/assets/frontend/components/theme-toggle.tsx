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

import { Moon, Sun } from "lucide-react"
import { useTheme } from "./theme-provider"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  
  const nextTheme = theme === "dark" ? "light" : "dark"
  const label = `Switch to ${nextTheme} theme (currently ${theme})`

  return (
    <button
      className="btn-icon relative focus-visible:ring-2 focus-visible:ring-nvidia-green focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-lg"
      onClick={() => setTheme(nextTheme)}
      aria-label={label}
      title={`Switch to ${nextTheme} theme`}
    >
      <Sun
        className={`h-5 w-5 transition-all ${theme === "dark" ? "opacity-0 scale-0 rotate-90 absolute" : "opacity-100 scale-100 rotate-0 relative"}`}
      />
      <Moon
        className={`h-5 w-5 transition-all ${theme === "light" ? "opacity-0 scale-0 -rotate-90 absolute" : "opacity-100 scale-100 rotate-0 relative"}`}
      />
    </button>
  )
}

