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
Download and process the marianna13/biorxiv dataset for txt2kg demo.
Filter for Creative Commons licensed papers and create individual txt files.
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
    print("Loading marianna13/biorxiv dataset...")
    
    # Load the dataset
    ds = load_dataset("marianna13/biorxiv")
    
    # Get the train split
    train_data = ds['train']
    
    print(f"Total dataset size: {len(train_data)} papers")
    
    # Filter for Creative Commons licensed papers
    cc_papers = train_data.filter(lambda x: x['LICENSE'] == 'creative-commons')
    
    print(f"Found {len(cc_papers)} Creative Commons licensed papers ({len(cc_papers)/len(train_data)*100:.1f}%)")
    
    # Take a sample for the demo (full dataset would be too large)
    sample_size = min(1000, len(cc_papers))  # Limit to 1000 papers for demo
    cc_sample = cc_papers.select(range(sample_size))
    
    print(f"Using sample of {len(cc_sample)} papers for demo")
    
    # Create output directory
    output_dir = Path("biorxiv_creative_commons")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating txt files in {output_dir}/")
    
    # Process each paper
    for i, item in enumerate(cc_sample):
        # Create filename from title and DOI
        title_part = sanitize_filename(item['TITLE'], max_length=50)
        doi_part = item['DOI'].replace('/', '_').replace('.', '_')
        filename = f"{i+1:03d}_{title_part}_{doi_part}.txt"
        
        # Create file content with full text
        content = f"Title: {item['TITLE']}\n"
        content += f"DOI: {item['DOI']}\n"
        content += f"Year: {item['YEAR']}\n"
        content += f"Authors: {'; '.join(item['AUTHORS']) if item['AUTHORS'] else 'N/A'}\n"
        content += f"License: {item['LICENSE']}\n"
        content += f"\nFull Text:\n{item['TEXT']}\n"
        
        # Write to file
        file_path = output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Successfully created {len(cc_sample)} txt files in {output_dir}/")
    
    # Show some statistics
    years = [item['YEAR'] for item in cc_sample]
    year_range = f"{min(years)} - {max(years)}"
    
    print(f"\nDataset Statistics:")
    print(f"  Year range: {year_range}")
    print(f"  License: Creative Commons (commercial use allowed)")
    print(f"  Content: Full paper text (not just abstracts)")
    print(f"  Average text length: {sum(len(item['TEXT']) for item in cc_sample) // len(cc_sample):,} characters")

if __name__ == "__main__":
    main()
