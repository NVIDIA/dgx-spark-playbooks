#!/usr/bin/env python3
#
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
#
"""
Download and process the MTEB raw_biorxiv dataset for txt2kg demo.
Filter for genetics/genomics categories and create individual txt files.
"""

import os
import re
from pathlib import Path
from datasets import load_dataset

def sanitize_filename(text, max_length=100):
    """Convert text to a safe filename."""
    # Remove special characters and replace with underscores
    filename = re.sub(r'[^\w\s-]', '', text)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename.strip('_')
    
    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    return filename

def main():
    print("Loading MTEB raw_biorxiv dataset...")
    
    # Load the dataset
    ds = load_dataset("mteb/raw_biorxiv")
    
    # Get the train split
    train_data = ds['train']
    
    print(f"Total dataset size: {len(train_data)} papers")
    
    # Filter for genetics or genomics categories
    genetics_genomics_data = []
    for item in train_data:
        category = item['category'].lower()
        if 'genetic' in category or 'genomic' in category:
            genetics_genomics_data.append(item)
    
    print(f"Found {len(genetics_genomics_data)} papers with genetics/genomics categories")
    
    if len(genetics_genomics_data) == 0:
        # Let's check what categories are available
        categories = set(item['category'] for item in train_data)
        print("Available categories:")
        for cat in sorted(categories):
            print(f"  - {cat}")
        return
    
    # Create output directory
    output_dir = Path("biorxiv_genetics_genomics")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating txt files in {output_dir}/")
    
    # Process each paper
    for i, item in enumerate(genetics_genomics_data):
        # Create filename from title and ID
        title_part = sanitize_filename(item['title'], max_length=50)
        paper_id = item['id'].replace('/', '_')
        filename = f"{i+1:03d}_{title_part}_{paper_id}.txt"
        
        # Create file content
        content = f"Title: {item['title']}\n"
        content += f"ID: {item['id']}\n"
        content += f"Category: {item['category']}\n"
        content += f"\nAbstract:\n{item['abstract']}\n"
        
        # Write to file
        file_path = output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Successfully created {len(genetics_genomics_data)} txt files in {output_dir}/")
    
    # Show some statistics
    categories_found = set(item['category'] for item in genetics_genomics_data)
    print(f"\nCategories included:")
    for cat in sorted(categories_found):
        count = sum(1 for item in genetics_genomics_data if item['category'] == cat)
        print(f"  - {cat}: {count} papers")

if __name__ == "__main__":
    main()
