# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

general_arguments = [
    'seed',
    'reproducibility',
    'state',
    'model',
    'data_path',
    'train_path',
    'checkpoint_dir',
    'show_progress',
    'config_file',
    'log_wandb',
    'use_text',
    'strategy',
    'precision'
]

training_arguments = [
    'epochs', 'train_batch_size',
    'optim_args',
    'eval_step', 'stopping_step',
    'clip_grad_norm',
    'loss_decimal_place',
    'gradient_accumulation_steps',
]

evaluation_arguments = [
    'eval_type',
    'repeatable',
    'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',
    'eval_batch_size',
    'metric_decimal_place',
]

dataset_arguments = [
    'MAX_TEXT_LENGTH',
    'MAX_ITEM_LIST_LENGTH',
    'MAX_ITEM_LIST_LENGTH_TEST',
    'MAX_GEN_TEXT_LENGTH',
    'MAX_USER_PROFILE_TEXT_LENGTH',
    'num_negatives',
    'text_keys',
    'item_prompt',
]
