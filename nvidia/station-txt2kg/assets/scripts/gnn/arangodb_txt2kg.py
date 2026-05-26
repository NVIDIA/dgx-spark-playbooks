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

import argparse
import gc
import json
import os
import torch
from glob import glob
from itertools import chain
from tqdm import tqdm
from python_arango import ArangoClient

# Import the necessary modules from PyTorch Geometric
from torch_geometric import seed_everything
from torch_geometric.nn import SentenceTransformer
from torch_geometric.utils.rag.backend_utils import (
    create_remote_backend_from_triplets,
    make_pcst_filter,
    preprocess_triplet,
)
from torch_geometric.utils.rag.feature_store import ModernBertFeatureStore
from torch_geometric.utils.rag.graph_store import NeighborSamplingRAGGraphStore
from torch_geometric.loader import RAGQueryLoader

# Define constants for better readability
NV_NIM_MODEL_DEFAULT = "nvidia/llama-3.1-nemotron-70b-instruct"
CHUNK_SIZE_DEFAULT = 512
DEFAULT_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1"

# ArangoDB defaults from docker-compose.yml
ARANGO_URL_DEFAULT = "http://localhost:8529"
ARANGO_DB_DEFAULT = "txt2kg"
ARANGO_USER_DEFAULT = ""
ARANGO_PASSWORD_DEFAULT = ""

# File paths and directories
DATASET_FILE = "tech_qa.pt"
TRIPLES_FILE = "tech_qa_just_triples.pt"
CHECKPOINT_FILE = "checkpoint_kg.pt"
TRAIN_DATA_FILE = "train.json"
CORPUS_DIR = "corpus"
BACKEND_PATH = "backend"
OUTPUT_DIR = "output"

def parse_args():
    parser = argparse.ArgumentParser()
    # Data processing related arguments
    parser.add_argument('--NV_NIM_MODEL', type=str, default=NV_NIM_MODEL_DEFAULT, help="The NIM LLM to use for TXT2KG for LLMJudge")
    parser.add_argument('--NV_NIM_KEY', type=str, default="", help="NVIDIA API key")
    parser.add_argument(
        '--ENDPOINT_URL', type=str, default=DEFAULT_ENDPOINT_URL, help=
        "The URL hosting your model, in case you are not using the public NIM."
    )
    parser.add_argument(
        '--chunk_size', type=int, default=512, help="When splitting context documents for txt2kg,\
        the maximum number of characters per chunk.")
    parser.add_argument('--checkpointing', action="store_true")
    
    # Add ArangoDB-specific arguments
    parser.add_argument('--arango_url', type=str, default=ARANGO_URL_DEFAULT, help="ArangoDB URL")
    parser.add_argument('--arango_db', type=str, default=ARANGO_DB_DEFAULT, help="ArangoDB database name")
    parser.add_argument('--arango_user', type=str, default=ARANGO_USER_DEFAULT, help="ArangoDB username")
    parser.add_argument('--arango_password', type=str, default=ARANGO_PASSWORD_DEFAULT, help="ArangoDB password")
    parser.add_argument('--use_arango', action="store_true", help="Use ArangoDB instead of TXT2KG")
    
    # Add file path arguments
    parser.add_argument('--dataset_file', type=str, default=DATASET_FILE, help="Path to save/load dataset")
    parser.add_argument('--triples_file', type=str, default=TRIPLES_FILE, help="Path to save/load triples")
    parser.add_argument('--checkpoint_file', type=str, default=CHECKPOINT_FILE, help="Path to save/load checkpoint")
    parser.add_argument('--train_data_file', type=str, default=TRAIN_DATA_FILE, help="Path to training data file")
    parser.add_argument('--corpus_dir', type=str, default=CORPUS_DIR, help="Directory containing corpus documents")
    parser.add_argument('--backend_path', type=str, default=BACKEND_PATH, help="Path for backend storage")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help="Directory for output files")
    
    return parser.parse_args()

def load_triples_from_arangodb(arango_url, arango_db, arango_user, arango_password):
    """
    Load triples from ArangoDB for use with the TXT2KG dataset
    
    Args:
        arango_url: ArangoDB connection URL
        arango_db: ArangoDB database name
        arango_user: ArangoDB username
        arango_password: ArangoDB password
        
    Returns:
        Array of triples in the format expected by create_remote_backend_from_triplets
    """
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts=arango_url)
        
        # Get database (no auth in our docker setup)
        if arango_user and arango_password:
            db = client.db(arango_db, username=arango_user, password=arango_password)
        else:
            db = client.db(arango_db)
        
        # Query to get all triples from ArangoDB as structured objects
        # Handle case sensitivity and trim whitespace
        aql_query = """
        FOR e IN relationships
        LET subject = TRIM(DOCUMENT(e._from).name)
        LET object = TRIM(DOCUMENT(e._to).name)
        LET predicate = TRIM(e.type)
        FILTER subject != "" AND predicate != "" AND object != ""
        RETURN {
            subject: subject,
            predicate: predicate,
            object: object
        }
        """
        
        # Execute the query
        cursor = db.aql.execute(aql_query)
        triple_dicts = list(cursor)
        
        # Format triples as strings in the format expected by PyTorch Geometric
        # The expected format is a list of strings in the form "subject predicate object"
        triples = format_triples_for_pytorch_geometric(triple_dicts)
        
        print(f"Loaded {len(triples)} triples from ArangoDB")
        # Print sample triples for debugging
        if len(triples) > 0:
            print("Sample triples:")
            for i in range(min(3, len(triples))):
                print(f"  {triples[i]}")
        
        return triples
    except Exception as error:
        print(f"Error loading triples from ArangoDB: {error}")
        raise error

def format_triples_for_pytorch_geometric(triple_dicts):
    """
    Format triples from ArangoDB into the format expected by PyTorch Geometric
    
    Args:
        triple_dicts: List of dictionaries with subject, predicate, object keys
        
    Returns:
        List of strings in the format "subject predicate object"
    """
    triples = []
    # Create a set to avoid duplicates
    unique_triples = set()
    
    for triple_dict in triple_dicts:
        # Skip any triple with empty values
        if not triple_dict['subject'] or not triple_dict['predicate'] or not triple_dict['object']:
            continue
            
        # Create a space-separated string in the format that preprocess_triplet expects
        triple_str = f"{triple_dict['subject']} {triple_dict['predicate']} {triple_dict['object']}"
        
        # Only add if not already in the set
        if triple_str not in unique_triples:
            unique_triples.add(triple_str)
            triples.append(triple_str)
    
    return triples

def get_data(args):
    # need a JSON dict of Questions and answers, see below for how its used
    with open(args.train_data_file) as file:
        json_obj = json.load(file)
    
    text_contexts = []
    # need a folder of text files to use for RAG and to make a KG from
    for file_path in glob(f"{args.corpus_dir}/*"):
        with open(file_path, "r+") as f:
            text_contexts.append(f.read())
    
    return json_obj, text_contexts

def validate_triple_format(triples):
    """
    Validate and fix triple format if needed to ensure compatibility with preprocess_triplet
    
    Args:
        triples: List of triples to validate
        
    Returns:
        Fixed list of triples in the format expected by preprocess_triplet
    """
    validated_triples = []
    
    print(f"Validating {len(triples)} triples...")
    for i, triple in enumerate(triples):
        # If triple is already a proper string with subject, predicate, object
        if isinstance(triple, str):
            parts = triple.split()
            # Ensure there are at least 3 parts (subject, predicate, object)
            if len(parts) >= 3:
                # For strings with more than 3 parts, use first as subject, second as predicate, 
                # and join the rest as object
                subject = parts[0]
                predicate = parts[1]
                obj = ' '.join(parts[2:])
                validated_triple = f"{subject} {predicate} {obj}"
                validated_triples.append(validated_triple)
            else:
                print(f"Warning: Triple at index {i} has fewer than 3 parts: {triple}")
        # If triple is a dictionary with subject, predicate, object keys
        elif isinstance(triple, dict) and 'subject' in triple and 'predicate' in triple and 'object' in triple:
            validated_triple = f"{triple['subject']} {triple['predicate']} {triple['object']}"
            validated_triples.append(validated_triple)
        # If triple is a tuple or list of length 3
        elif (isinstance(triple, tuple) or isinstance(triple, list)) and len(triple) == 3:
            validated_triple = f"{triple[0]} {triple[1]} {triple[2]}"
            validated_triples.append(validated_triple)
        else:
            print(f"Warning: Skipping triple at index {i} with invalid format: {triple}")
    
    print(f"Validation complete. {len(validated_triples)} valid triples out of {len(triples)}")
    return validated_triples

def make_dataset(args):
    """Modified make_dataset function that can use ArangoDB as a data source"""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_path = os.path.join(args.output_dir, args.dataset_file)
    triples_path = os.path.join(args.output_dir, args.triples_file)
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_file)
    
    if os.path.exists(dataset_path):
        print(f"Re-using Saved TechQA KG-RAG Dataset from {dataset_path}...")
        return torch.load(dataset_path, weights_only=False)
    else:
        qa_pairs, context_docs = get_data(args)
        print("Number of Docs in our VectorDB =", len(context_docs))
        data_lists = {"train": [], "validation": [], "test": []}
        
        # Load triples either from saved file or from sources
        triples = []
        if os.path.exists(triples_path):
            triples = torch.load(triples_path, weights_only=False)
        else:
            if args.use_arango:
                # Load triples from ArangoDB instead of generating with TXT2KG
                print("Loading triples from ArangoDB...")
                triples = load_triples_from_arangodb(
                    args.arango_url, 
                    args.arango_db, 
                    args.arango_user, 
                    args.arango_password
                )
                # Validate and fix triples format if needed
                triples = validate_triple_format(triples)
                # Save triples for future use
                torch.save(triples, triples_path)
            else:
                # Original TXT2KG code path
                from torch_geometric.nn import TXT2KG
                kg_maker = TXT2KG(
                    NVIDIA_NIM_MODEL=args.NV_NIM_MODEL,
                    NVIDIA_API_KEY=args.NV_NIM_KEY,
                    ENDPOINT_URL=args.ENDPOINT_URL,
                    chunk_size=args.chunk_size
                )
                print(
                    "Note that if the TXT2KG process is too slow for you're liking using the public NIM, "
                    "consider deploying yourself using local_lm flag of TXT2KG or using "
                    "https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct?snippet_tab=Docker "
                    "to deploy to a private endpoint, which you can pass to this script w/ --ENDPOINT_URL flag."
                )
                
                total_tqdm_count = len(context_docs)
                initial_tqdm_count = 0
                if os.path.exists(checkpoint_path):
                    print(f"Restoring KG from checkpoint at {checkpoint_path}...")
                    saved_relevant_triples = torch.load(checkpoint_path, weights_only=False)
                    kg_maker.relevant_triples = saved_relevant_triples
                    kg_maker.doc_id_counter = len(saved_relevant_triples)
                    initial_tqdm_count = kg_maker.doc_id_counter
                    context_docs = context_docs[(kg_maker.doc_id_counter - 1):]
                
                if args.checkpointing:
                    interval = 10
                    count = 0
                
                for context_doc in tqdm(context_docs, total=total_tqdm_count, 
                                       initial=initial_tqdm_count, desc="Extracting KG triples"):
                    kg_maker.add_doc_2_KG(txt=context_doc)
                    if args.checkpointing:
                        count += 1
                        if count == interval:
                            print(f" checkpointing KG to {checkpoint_path}...")
                            count = 0
                            kg_maker.save_kg(checkpoint_path)
                
                relevant_triples = kg_maker.relevant_triples
                triples.extend(
                    list(
                        chain.from_iterable(
                            triple_set for triple_set in relevant_triples.values()
                        )
                    )
                )
                triples = list(dict.fromkeys(triples))
                torch.save(triples, triples_path)
                
                if args.checkpointing and os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
        
        print("Number of triples in our GraphDB =", len(triples))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sent_trans_batch_size = 256
        model = SentenceTransformer(
            model_name='Alibaba-NLP/gte-modernbert-base').to(device)
        
        backend_path = os.path.join(args.output_dir, args.backend_path)
        fs, gs = create_remote_backend_from_triplets(
            triplets=triples,
            node_embedding_model=model,
            node_method_to_call="encode",
            path=backend_path,
            pre_transform=preprocess_triplet,
            node_method_kwargs={
                "batch_size": min(len(triples), sent_trans_batch_size)
            },
            graph_db=NeighborSamplingRAGGraphStore,
            feature_db=ModernBertFeatureStore
        ).load()
        
        # encode the raw context docs
        embedded_docs = model.encode(
            context_docs,
            output_device=device,
            batch_size=int(sent_trans_batch_size / 4),
            verbose=True
        )
        
        # k for KNN
        knn_neighsample_bs = 1024
        # number of neighbors for each seed node selected by KNN
        fanout = 100
        # number of hops for neighborsampling
        num_hops = 2
        
        local_filter_kwargs = {
            "topk": 5,  # nodes
            "topk_e": 5,  # edges
            "cost_e": .5,  # edge cost
            "num_clusters": 10,  # num clusters
        }
        
        print("Now to retrieve context for each query from our Vector and Graph DBs...")
        # GraphDB retrieval done with KNN+NeighborSampling+PCST
        # PCST = Prize Collecting Steiner Tree
        # VectorDB retrieval just vanilla RAG
        query_loader = RAGQueryLoader(
            data=(fs, gs),
            seed_nodes_kwargs={"k_nodes": knn_neighsample_bs},
            sampler_kwargs={"num_neighbors": [fanout] * num_hops},
            local_filter=make_pcst_filter(triples, model),
            local_filter_kwargs=local_filter_kwargs,
            raw_docs=context_docs,
            embedded_docs=embedded_docs
        )
        
        total_data_list = []
        extracted_triple_sizes = []
        
        for data_point in tqdm(qa_pairs, desc="Building un-split dataset"):
            if data_point["is_impossible"]:
                continue
                
            QA_pair = (data_point["question"], data_point["answer"])
            q = QA_pair[0]
            subgraph = query_loader.query(q)
            subgraph.label = QA_pair[1]
            total_data_list.append(subgraph)
            extracted_triple_sizes.append(len(subgraph.triples))
        
        import random
        random.shuffle(total_data_list)
        
        print("Min # of Retrieved Triples =", min(extracted_triple_sizes))
        print("Max # of Retrieved Triples =", max(extracted_triple_sizes))
        print("Average # of Retrieved Triples =", sum(extracted_triple_sizes) / len(extracted_triple_sizes))
        
        # 60:20:20 split
        data_lists["train"] = total_data_list[:int(.6 * len(total_data_list))]
        data_lists["validation"] = total_data_list[
            int(.6 * len(total_data_list)):int(.8 * len(total_data_list))]
        data_lists["test"] = total_data_list[int(.8 * len(total_data_list)):]
        
        torch.save(data_lists, dataset_path)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return data_lists

if __name__ == '__main__':
    # for reproducibility
    seed_everything(50)
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process and save dataset
    data_lists = make_dataset(args)
    print(f"Dataset processed and saved to {os.path.join(args.output_dir, args.dataset_file)}")
    print("Training data size:", len(data_lists["train"]))
    print("Validation data size:", len(data_lists["validation"]))
    print("Testing data size:", len(data_lists["test"])) 