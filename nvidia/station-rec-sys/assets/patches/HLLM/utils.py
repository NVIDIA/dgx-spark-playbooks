# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import copy
import importlib
import os
import pickle
from logging import getLogger
from REC.data.dataset import *
from REC.utils import set_color
from functools import partial
from .dataload import Data
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
import copy


def load_data(config):
    dataload = Data(config)
    return dataload


def bulid_dataloader(config, dataload):
    '''
    split dataset, generate user history sequence, train/valid/test dataset
    '''
    dataset_dict = {
        'SASRec': ('SEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'HSTU': ('SEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'LLMIDRec': ('SEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'HLLM': (('TextSEQTrainDataset', 'customize_rmpad_collate'), 'SeqEvalDataset', 'seq_eval_collate'),
        'HLLM_Creator': (('TextSEQCreatorTrainDataset', 'customize_rmpad_collate'), None, None)
    }

    model_name = config['model']
    dataload.build()

    dataset_module = importlib.import_module('REC.data.dataset')
    train_set_name, test_set_name, collate_fn_name = dataset_dict[model_name]

    if isinstance(train_set_name, tuple):
        train_set_class = getattr(dataset_module, train_set_name[0])
        train_collate_fn = getattr(dataset_module, train_set_name[1])
    else:
        train_set_class = getattr(dataset_module, train_set_name)
        train_collate_fn = None

    train_data = train_set_class(config, dataload)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow')
    )

    num_workers = config.get('num_workers', 24)
    logger.info(f"DataLoader num_workers = {num_workers}, "
                  f"persistent_workers = True, prefetch_factor = 4") 
    rank = torch.distributed.get_rank()
    seed = torch.initial_seed()

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed
    )

    if train_collate_fn:                                                                                                                     
        train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], num_workers=num_workers,
                                pin_memory=True, sampler=train_sampler, collate_fn=train_collate_fn, worker_init_fn=init_fn,               
                                persistent_workers=True, prefetch_factor=4)                                                 
    else:                                                                                                                                    
        train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], num_workers=num_workers,
                                pin_memory=True, sampler=train_sampler, worker_init_fn=init_fn,                                            
                                persistent_workers=True, prefetch_factor=4)   

    if test_set_name is not None:
        test_set_class = getattr(dataset_module, test_set_name)
        eval_collate_fn = getattr(dataset_module, collate_fn_name)

        valid_data = test_set_class(config, dataload, phase='valid')
        test_data = test_set_class(config, dataload, phase='test')

        logger.info(
            set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
            set_color(f'[{config["eval_batch_size"]}]', 'yellow')
        )

        valid_sampler = NonConsecutiveSequentialDistributedSampler(valid_data)
        test_sampler = NonConsecutiveSequentialDistributedSampler(test_data)

        valid_loader = DataLoader(valid_data, batch_size=config['eval_batch_size'], num_workers=num_workers,
                              pin_memory=True, sampler=valid_sampler, collate_fn=eval_collate_fn,
                              persistent_workers=True, prefetch_factor=4)

        test_loader = DataLoader(test_data, batch_size=config['eval_batch_size'], num_workers=num_workers,
                                pin_memory=True, sampler=test_sampler, collate_fn=eval_collate_fn,
                                persistent_workers=True, prefetch_factor=4)
    else:
        valid_loader = test_loader = None

    return train_loader, valid_loader, test_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    worker_seed = initial_seed + worker_id + torch.distributed.get_rank()
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class NonConsecutiveSequentialDistributedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(self.dataset)
        self.num_samples = math.ceil(
            (self.total_size-self.rank)/self.num_replicas
        )

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class ConsecutiveSequentialDistributedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
