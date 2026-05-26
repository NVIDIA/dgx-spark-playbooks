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
import { useState, useCallback, useRef } from 'react'

interface UseShiftSelectOptions<T> {
  items: T[]
  getItemId: (item: T) => string
  canSelect?: (item: T) => boolean
  onSelectionChange?: (selectedIds: string[]) => void
}

interface UseShiftSelectReturn<T> {
  selectedItems: string[]
  setSelectedItems: (items: string[]) => void
  handleItemClick: (item: T, event?: React.MouseEvent) => void
  handleSelectAll: () => void
  isSelected: (itemId: string) => boolean
  clearSelection: () => void
}

export function useShiftSelect<T>({
  items,
  getItemId,
  canSelect = () => true,
  onSelectionChange
}: UseShiftSelectOptions<T>): UseShiftSelectReturn<T> {
  const [selectedItems, setSelectedItemsState] = useState<string[]>([])
  const lastClickedIndexRef = useRef<number | null>(null)

  const setSelectedItems = useCallback((items: string[]) => {
    setSelectedItemsState(items)
    onSelectionChange?.(items)
  }, [onSelectionChange])

  const isSelected = useCallback((itemId: string) => {
    return selectedItems.includes(itemId)
  }, [selectedItems])

  const clearSelection = useCallback(() => {
    setSelectedItems([])
    lastClickedIndexRef.current = null
  }, [setSelectedItems])

  const handleItemClick = useCallback((item: T, event?: React.MouseEvent) => {
    if (!canSelect(item)) return

    const itemId = getItemId(item)
    const currentIndex = items.findIndex(i => getItemId(i) === itemId)
    
    if (currentIndex === -1) return

    // Handle shift+click for range selection
    if (event?.shiftKey && lastClickedIndexRef.current !== null) {
      const lastIndex = lastClickedIndexRef.current
      const start = Math.min(currentIndex, lastIndex)
      const end = Math.max(currentIndex, lastIndex)
      
      // Get all selectable items in the range
      const rangeItems = items.slice(start, end + 1)
      const selectableRangeIds = rangeItems
        .filter(canSelect)
        .map(getItemId)
      
      // Add range to current selection (union)
      const newSelection = Array.from(new Set([...selectedItems, ...selectableRangeIds]))
      setSelectedItems(newSelection)
    }
    // Handle ctrl/cmd+click for individual toggle
    else if (event?.ctrlKey || event?.metaKey) {
      if (isSelected(itemId)) {
        setSelectedItems(selectedItems.filter(id => id !== itemId))
      } else {
        setSelectedItems([...selectedItems, itemId])
      }
      lastClickedIndexRef.current = currentIndex
    }
    // Handle regular click - toggle individual item
    else {
      if (isSelected(itemId)) {
        setSelectedItems(selectedItems.filter(id => id !== itemId))
      } else {
        setSelectedItems([...selectedItems, itemId])
      }
      lastClickedIndexRef.current = currentIndex
    }
  }, [items, selectedItems, canSelect, getItemId, isSelected, setSelectedItems])

  const handleSelectAll = useCallback(() => {
    const selectableItems = items.filter(canSelect)
    const selectableIds = selectableItems.map(getItemId)
    
    // If all selectable items are selected, deselect all
    if (selectedItems.length === selectableIds.length && 
        selectableIds.every(id => selectedItems.includes(id))) {
      setSelectedItems([])
      lastClickedIndexRef.current = null
    } else {
      // Otherwise, select all selectable items
      setSelectedItems(selectableIds)
      lastClickedIndexRef.current = null
    }
  }, [items, selectedItems, canSelect, getItemId, setSelectedItems])

  return {
    selectedItems,
    setSelectedItems,
    handleItemClick,
    handleSelectAll,
    isSelected,
    clearSelection
  }
}
