"""HLLM item-embedding extraction.

By default this script does ONLY the work strictly required to produce
$DATA_DIR/processed/hllm_item_embeddings.npy and hllm_item_id_map.npy:
forward-pass every item text through the trained item LLM via HLLM's
compute_item_feature(), then dump the resulting (N, hidden_dim) matrix to
disk. Downstream code (FAISS, the UI, the re-ranker) consumes the .npy file
and never re-loads the HLLM model. Wall time on a single GB300: ~30-60s.

An optional --regression-eval flag re-runs HLLM's validation and test metric
loops on the loaded checkpoint and prints R@K / NDCG@K. Use it as a
regression gate before promoting a newly-trained retriever to production:
compare the printed numbers against the previous retriever's baseline. The
flag adds ~3-5 min on a single GB300 (two held-out passes plus a redundant
item-feature recompute inside the second pass).

USAGE
  # Fast (default) — just produce embeddings.
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 extract_embeddings.py

  # Production regression check.
  ... extract_embeddings.py --regression-eval
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

WORKSPACE = os.environ.get('PLAYBOOK_WORKSPACE', os.path.expanduser('~'))
HLLM_CODE_DIR = os.path.join(WORKSPACE, 'hllm-code')
DATA_DIR = os.path.join(WORKSPACE, 'data')
MODELS_DIR = os.path.join(WORKSPACE, 'models')
CHECKPOINTS_DIR = os.path.join(WORKSPACE, 'checkpoints')
sys.path.insert(0, HLLM_CODE_DIR)

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy

from REC.data import load_data, bulid_dataloader
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer

# Config
DATASET = 'amazon_dresses'
PRETRAIN_DIR = os.path.join(MODELS_DIR, 'TinyLlama-1.1B')
CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, 'dresses_lora_r16')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')


def checkpoint_dir_from_path(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    if path.name == 'mp_rank_00_model_states.pt' and path.parent.name == 'checkpoint':
        return str(path.parent.parent)
    if path.name == 'checkpoint' and path.parent.name.endswith('.pth'):
        return str(path.parent)
    if path.name.endswith('.pth'):
        return str(path)
    return ckpt_path


def init_and_load(trainer, config, checkpoint_file):
    # Mirrors the setup + load portion of trainer.evaluate() (HLLM
    # trainer.py lines ~615-642). Lets the fast path call
    # compute_item_feature() directly without paying the metrics-loop cost.
    world_size = int(os.environ['WORLD_SIZE'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    nnodes = world_size // local_world_size
    precision = config['precision'] if config['precision'] else '32'

    if config['strategy'] == 'deepspeed':
        strategy = DeepSpeedStrategy(
            stage=config['stage'],
            precision=precision,
            exclude_frozen_parameters=config.get('exclude_frozen_parameters', True),
        )
        trainer.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        trainer.lite.launch()
        trainer.model, trainer.optimizer = trainer.lite.setup(trainer.model, trainer.optimizer)
    else:
        strategy = DDPStrategy(find_unused_parameters=True)
        trainer.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        trainer.lite.launch()
        trainer.model = trainer.lite.setup(trainer.model)

    state = {"model": trainer.model}
    trainer.lite.load(checkpoint_file, state, strict=False)


def main():
    parser = argparse.ArgumentParser(
        description='Extract HLLM item embeddings, with an optional regression-eval pass.'
    )
    parser.add_argument(
        '--ckpt_path',
        default=os.path.join(CHECKPOINT_DIR, 'HLLM-0.pth', 'checkpoint', 'mp_rank_00_model_states.pt'),
    )
    parser.add_argument('--output_dir', default=OUTPUT_DIR)
    parser.add_argument(
        '--regression-eval', dest='regression_eval', action='store_true',
        help='Also run HLLM validation + test metric loops on the loaded '
             'checkpoint to gate regressions against the prior baseline. '
             'Adds ~3-5 min on a single GB300.',
    )
    args = parser.parse_args()

    checkpoint_model_file = checkpoint_dir_from_path(args.ckpt_path)
    checkpoint_config_dir = (
        str(Path(checkpoint_model_file).parent)
        if Path(checkpoint_model_file).name.endswith('.pth')
        else checkpoint_model_file
    )

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    config_files = [
        os.path.join(HLLM_CODE_DIR, 'overall', 'LLM_deepspeed.yaml'),
        os.path.join(HLLM_CODE_DIR, 'HLLM', 'HLLM.yaml'),
    ]
    config = Config(config_file_list=config_files)
    config['device'] = torch.device('cuda', local_rank)

    overrides = {
        'dataset': DATASET,
        'data_path': os.path.join(HLLM_CODE_DIR, 'dataset') + '/',
        'text_path': os.path.join(HLLM_CODE_DIR, 'information', DATASET + '.csv'),
        'text_keys': ['title', 'description'],
        'item_pretrain_dir': PRETRAIN_DIR,
        'user_pretrain_dir': PRETRAIN_DIR,
        'item_llm_init': True,
        'user_llm_init': True,
        'gradient_checkpointing': False,
        'train_batch_size': 32,
        'MAX_ITEM_LIST_LENGTH': 20,
        'MAX_TEXT_LENGTH': 64,
        'loss': 'nce',
        'lora_r': 16,
        'lora_alpha': 64,
        'lora_dropout': 0.0,
        'lora_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
        'checkpoint_dir': checkpoint_config_dir,
        'show_progress': True,
        'log_wandb': False,
    }
    for k, v in overrides.items():
        config.final_config_dict[k] = v

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    print("=" * 60)
    print("EXTRACT TRAINED HLLM EMBEDDINGS")
    print("=" * 60)
    print(f"Checkpoint:        {args.ckpt_path}")
    print(f"Resolved ckpt dir: {checkpoint_model_file}")
    print(f"Output dir:        {args.output_dir}")
    if args.regression_eval:
        print("Mode:              embeddings + regression eval (valid + test metrics)")
    else:
        print("Mode:              embeddings only — pass --regression-eval to also run")
        print("                   HLLM's held-out metric loops as a regression check.")
    t0 = time.time()

    # Build dataloaders + model — needed for both fast and regression paths.
    dataload = load_data(config)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
    print(f"Items: {dataload.item_num:,}, Users: {dataload.user_num:,}")

    model = get_model(config['model'])(config, dataload)
    trainer = Trainer(config, model)

    # trainer.evaluate(valid_loader, init_model=True, load_best_model=True)
    # bundles four things: Fabric/DeepSpeed init, checkpoint load,
    # compute_item_feature() (the embedding side effect we need), and the
    # metrics loop. Regression mode lets evaluate() do all four. Default
    # mode replicates the first three via init_and_load() + a direct
    # compute_item_feature() call, skipping the metrics loop.
    if args.regression_eval:
        print("\nValidation eval (regression check)...")
        valid_result = trainer.evaluate(
            valid_loader,
            load_best_model=True,
            model_file=checkpoint_model_file,
            show_progress=True,
            init_model=True,
        )
        print(f"Validation: {valid_result}")
    else:
        print("\nLoading checkpoint and computing item embeddings...")
        init_and_load(trainer, config, checkpoint_model_file)
        with torch.no_grad():
            trainer.model.eval()
            trainer.tot_item_num = dataload.item_num
            trainer.compute_item_feature(config, dataload)

    # Save item embeddings (always).
    item_feature = trainer.item_feature
    if isinstance(item_feature, tuple):
        item_feature = item_feature[0]

    item_embeddings = item_feature.cpu().float().numpy()
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    item_embeddings = item_embeddings / np.maximum(norms, 1e-8)

    print(f"\nItem embeddings shape: {item_embeddings.shape}")

    # Item 0 is padding in HLLM, real items are 1-indexed.
    item_id_map = getattr(dataload, 'id2token', None)
    if isinstance(item_id_map, dict):
        item_id_map = item_id_map.get('item_id')
    if item_id_map is None:
        item_text = pd.read_csv(os.path.join(HLLM_CODE_DIR, 'information', f'{DATASET}.csv'))
        item_id_map = ['[PAD]'] + item_text['item_id'].tolist()
        print(f"Item ID map from CSV: {len(item_id_map)} entries (including padding)")
    else:
        item_id_map = np.asarray(item_id_map, dtype=str)
        print(f"Item ID map from HLLM dataload: {len(item_id_map)} entries")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'hllm_item_embeddings.npy', item_embeddings)
    np.save(output_dir / 'hllm_item_id_map.npy', np.asarray(item_id_map, dtype=str))
    print(f"Saved embeddings to {args.output_dir}/")

    # Test eval — only as part of the regression-check pass. Setup +
    # checkpoint already in place from the validation evaluate() above.
    if args.regression_eval:
        print("\nTest eval (unbiased generalization number)...")
        test_result = trainer.evaluate(
            test_loader,
            load_best_model=False,
            show_progress=True,
        )
        print(f"Test: {test_result}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
