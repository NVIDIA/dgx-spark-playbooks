# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import os
import sys
import math
from logging import getLogger
from time import time
import time as t
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import deepspeed
from deepspeed.ops.adam import FusedAdam

from REC.data.dataset import BatchTextDataset
from REC.data.dataset.collate_fn import customize_rmpad_collate
from torch.utils.data import DataLoader
from REC.evaluator import Evaluator, Collector
from REC.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    get_tensorboard, set_color, get_gpu_usage, WandbLogger
from REC.utils.lr_scheduler import *

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy

# torch.compile: allow more recompiles for variable packed-sequence shapes
# (HLLM's collate uses cu_input_lens → shapes differ per batch)
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.capture_scalar_outputs = True  # trace through .item() calls (e.g. flash-attn max_seqlen)


class Trainer(object):
    def __init__(self, config, model):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.logger = getLogger()

        self.wandblogger = WandbLogger(config)

        self.optim_args = config['optim_args']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.max_steps = config.get('max_steps', 0)  # 0 = unlimited (use epochs)
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']

        self.rank = torch.distributed.get_rank()

        if self.rank == 0:
            self.tensorboard = get_tensorboard(self.logger)

        self.checkpoint_dir = config['checkpoint_dir']
        if self.rank == 0:
            ensure_dir(self.checkpoint_dir)

        self.saved_model_name = '{}-{}.pth'.format(self.config['model'], 0)
        self.saved_model_file = os.path.join(self.checkpoint_dir, self.saved_model_name)

        self.use_text = config['use_text']

        self.start_epoch = 0
        self.cur_step = 0
        self.global_step_count = 0  # tracks steps across epochs for max_steps
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.update_interval = config['update_interval'] if config['update_interval'] else 20
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        self.scheduler_config = config['scheduler_args']
        if config['freeze_prefix'] or config['freeze_ad']:
            freeze_prefix = config['freeze_prefix'] if config['freeze_prefix'] else []
            if config['freeze_ad']:
                freeze_prefix.extend(['item_llm', 'item_emb_tokens'])
            if not config['ft_item']:
                freeze_prefix.extend(['item_embedding'])

            self._freeze_params(freeze_prefix)

        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()} {p.requires_grad}")

        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_feature = None
        self.tot_item_num = None

    def _freeze_params(self, freeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.logger.info(f"freeze_params: {name}")
                    param.requires_grad = False

    def _build_scheduler(self, warmup_steps=None, tot_steps=None):
        if self.scheduler_config['type'] == 'cosine':
            self.logger.info(f"Use consine scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        elif self.scheduler_config['type'] == 'liner':
            self.logger.info(f"Use linear scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        else:
            self.logger.info(f"Use constant scheduler")
            return get_constant_schedule(self.optimizer)

    def _build_optimizer(self):
        if len(self.optim_args) == 4:
            params = self.model.named_parameters()
            modal_params = []
            recsys_params = []
            modal_decay_params = []
            recsys_decay_params = []
            decay_check_name = self.config['decay_check_name']
            for index, (name, param) in enumerate(params):
                if param.requires_grad:
                    if 'visual_encoder' in name:
                        modal_params.append(param)
                    else:
                        recsys_params.append(param)
                    if decay_check_name:
                        if decay_check_name in name:
                            modal_decay_params.append(param)
                        else:
                            recsys_decay_params.append(param)
            if decay_check_name:
                optimizer = optim.AdamW([
                    {'params': modal_decay_params, 'lr': self.optim_args['modal_lr'], 'weight_decay': self.optim_args['modal_decay']},
                    {'params': recsys_decay_params, 'lr': self.optim_args['rec_lr'], 'weight_decay': self.optim_args['rec_decay']}
                ])
                optim_output = set_color(f'recsys_decay_params_len: {len(recsys_decay_params)}  modal_params_decay_len: {len(modal_decay_params)}', 'blue')
                self.logger.info(optim_output)
            else:
                optimizer = optim.AdamW([
                    {'params': modal_params, 'lr': self.optim_args['modal_lr'], 'weight_decay': self.optim_args['modal_decay']},
                    {'params': recsys_params, 'lr': self.optim_args['rec_lr'], 'weight_decay': self.optim_args['rec_decay']}
                ])
                optim_output = set_color(f'recsys_lr_params_len: {len(recsys_params)}  modal_lr_params_len: {len(modal_params)}', 'blue')
                self.logger.info(optim_output)
        elif self.config['lr_mult_prefix'] and self.config['lr_mult_rate']:
            normal_params_dict = {
                "params": [],
                "lr": self.optim_args['learning_rate'],
                "weight_decay": self.optim_args['weight_decay']
            }
            high_lr_params_dict = {
                "params": [],
                "lr": self.optim_args['learning_rate'] * self.config['lr_mult_rate'],
                "weight_decay": self.optim_args['weight_decay']
            }
            self.logger.info(f'Use higher lr rate {self.config["lr_mult_rate"]} x {self.optim_args["learning_rate"]} for prefix {self.config["lr_mult_prefix"]}')

            for n, p in self.model.named_parameters():
                if any(n.startswith(x) for x in self.config['lr_mult_prefix']):
                    self.logger.info(f"high lr param: {n} {self.optim_args['learning_rate'] * self.config['lr_mult_rate']}")
                    high_lr_params_dict["params"].append(p)
                else:
                    normal_params_dict["params"].append(p)
            optimizer = optim.AdamW([normal_params_dict, high_lr_params_dict])
        elif self.config['optimizer_kwargs']:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.config['optimizer_kwargs']['optimizer']['params']['lr'] = self.optim_args['learning_rate']
            self.config['optimizer_kwargs']['optimizer']['params']['weight_decay'] = self.optim_args['weight_decay']
            optimizer = deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam(params, **self.config['optimizer_kwargs']['optimizer']['params'])
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            # use deepspeed fused adam optimizer if set in the config
            if self.config.get('use_fused_adam', True):
                optimizer = FusedAdam(
                    params,
                    lr=self.optim_args['learning_rate'],
                    weight_decay=self.optim_args['weight_decay'],
                    adam_w_mode=True,
                )
                self.logger.info(
                    f"Optimizer: DeepSpeed FusedAdam (GPU fused kernels), "
                    f"adam_w_mode=True, lr={self.optim_args['learning_rate']}, "
                    f"weight_decay={self.optim_args['weight_decay']}"
                )
            # otherwise just use AdamW
            else:
                optimizer = optim.AdamW(
                    params,
                    lr=self.optim_args['learning_rate'],
                    weight_decay=self.optim_args['weight_decay'],
                )
                self.logger.info("Optimizer: torch.optim.AdamW (fused_adam disabled)")
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, show_progress=False, valid_data=None):
        self.model.train()
        total_loss = 0
        # auto_resume: skip already-completed batches on the first resumed epoch.
        # Cleared after this epoch so subsequent epochs run full-length.
        skip_batches = 0
        if epoch_idx == self.start_epoch and getattr(self, '_resume_batch_idx', 0) > 0:
            skip_batches = self._resume_batch_idx
            if self.rank == 0:
                self.logger.info(
                    f"auto_resume: skipping first {skip_batches} batches of epoch {epoch_idx}"
                )
            self._resume_batch_idx = 0
        if self.rank == 0:
            pbar = tqdm(
                total=len(train_data),
                miniters=self.update_interval,
                desc=set_color(f"Train [{epoch_idx:>3}/{self.epochs:>3}]", 'pink'),
                file=sys.stdout
            )
        accum_steps = self.grad_accum_steps
        grad_norm = None
        bwd_time = t.time()
        self.optimizer.zero_grad()
        for batch_idx, data in enumerate(train_data):
            if batch_idx < skip_batches:
                continue
            start_time = bwd_time
            data = self.to_device(data)
            data_time = t.time()
            losses = self.model(data)
            fwd_time = t.time()
            if self.config['loss'] and self.config['loss'] == 'nce':
                model_out = losses
                losses = model_out.pop('loss')
            self._check_nan(losses)
            total_loss = total_loss + losses.item()
            self.lite.backward(losses / accum_steps)
            is_accum_step = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_data)
            if is_accum_step:
                grad_norm = self.optimizer.step()
                self.optimizer.zero_grad()
                bwd_time = t.time()
                if self.scheduler_config:
                    self.lr_scheduler.step()
            else:
                bwd_time = t.time()
            # Step-based checkpoint saving (counted in micro-batches)
            global_step = epoch_idx * len(train_data) + batch_idx
            save_steps = self.config.get('save_steps', 0)
            if save_steps > 0 and batch_idx > 0 and batch_idx % save_steps == 0:
                step_ckpt_name = '{}-{}-step-{}.pth'.format(self.config['model'], epoch_idx, batch_idx)
                step_state = {
                    "model": self.model,
                    "optimizer": self.optimizer,
                    'config': self.config,
                    'epoch': epoch_idx,
                    'batch_idx': batch_idx,
                    'cur_step': self.cur_step,
                    'best_valid_score': self.best_valid_score,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state()
                }
                self.lite.save(os.path.join(self.checkpoint_dir, step_ckpt_name), state=step_state)
                if self.rank == 0:
                    self.logger.info(f"Step checkpoint saved: {step_ckpt_name}")
                # Keep only the latest N step checkpoints
                max_keep = self.config.get('max_keep_checkpoints', 3)
                if self.rank == 0 and max_keep > 0:
                    import glob, shutil
                    pattern = os.path.join(self.checkpoint_dir, '{}-*-step-*.pth'.format(self.config['model']))
                    existing = sorted(glob.glob(pattern), key=os.path.getmtime)
                    while len(existing) > max_keep:
                        old = existing.pop(0)
                        shutil.rmtree(old, ignore_errors=True)
                        self.logger.info(f"Removed old checkpoint: {os.path.basename(old)}")

            # Per-step W&B logging
            wandb_log_interval = self.config.get('wandb_log_interval', 100)
            if self.rank == 0 and wandb_log_interval > 0 and batch_idx % wandb_log_interval == 0:
                step_metrics = {
                    'step_loss': losses.item(),
                    'global_step': global_step,
                    'epoch_progress': batch_idx / len(train_data),
                }
                if self.scheduler_config:
                    step_metrics['learning_rate'] = self.lr_scheduler.get_lr()[0]
                if self.config['loss'] and self.config['loss'] == 'nce' and isinstance(model_out, dict):
                    for k, v in model_out.items():
                        step_metrics[k] = v.item() if hasattr(v, 'item') else v
                # Cross-run normalized metrics: NCE loss ceiling is log(N+1) where N = effective
                # negatives (random-guess baseline). Dividing by that puts any run on a 0->1 scale
                # (0 = perfect, 1 = random), so loss / top-k curves from different num_negatives
                # settings are directly comparable. Same idea for top-k lift over random = k/(N+1).
                nce_samples = step_metrics.get('nce_samples')
                if nce_samples is not None and nce_samples > 0:
                    rand_ceil = math.log(nce_samples + 1.0)
                    if rand_ceil > 0:
                        step_metrics['loss_over_random'] = step_metrics['step_loss'] / rand_ceil
                    for k in (1, 5, 10, 50, 100):
                        acc_key = f'nce_top{k}_acc'
                        if acc_key in step_metrics:
                            random_p = k / (nce_samples + 1.0)
                            if random_p > 0:
                                step_metrics[f'{acc_key}_lift'] = step_metrics[acc_key] / random_p
                self.wandblogger.log_metrics(step_metrics, head='train_step')

            # In-epoch full-catalog validation: expensive (one pass of _full_sort_batch_eval,
            # ~300 s on dresses), but the only way to see paper-family Recall@K / NDCG@K before
            # the epoch boundary. Gated by `fast_eval_interval` (0 = disabled). Recommended: 500
            # on dresses (~15% overhead), higher on larger catalogs. Logged under `valid_fast/*`
            # so it doesn't overwrite the authoritative per-epoch `valid/*` curve.
            fast_eval_interval = self.config.get('fast_eval_interval', 0)
            if (fast_eval_interval > 0 and valid_data is not None
                    and batch_idx > 0 and batch_idx % fast_eval_interval == 0):
                torch.distributed.barrier()
                fast_start = t.time()
                fast_valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=False)
                torch.distributed.barrier()
                self.model.train()  # evaluate() sets model.eval(); flip back for training
                if self.rank == 0 and fast_valid_result:
                    fast_elapsed = t.time() - fast_start
                    self.logger.info(
                        f"fast eval @ step {global_step}: "
                        f"{dict2str(fast_valid_result)} (took {fast_elapsed:.1f}s)"
                    )
                    self.wandblogger.log_metrics(
                        {**fast_valid_result, 'global_step': global_step},
                        head='valid_fast'
                    )

            if show_progress and self.rank == 0 and batch_idx % self.update_interval == 0:
                msg = f"loss: {losses:.4f} data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f} bwd: {bwd_time-fwd_time:.3f}"
                if self.scheduler_config:
                    msg = f"lr: {self.lr_scheduler.get_lr()[0]:.7f} " + msg
                if self.config['loss'] and self.config['loss'] == 'nce':
                    for k, v in model_out.items():
                        if k.endswith('loss'):
                            msg += f" {k}: {v:.3f}"
                if grad_norm:
                    msg = msg + f" grad_norm: {grad_norm.sum():.4f}"
                pbar.set_postfix_str(msg, refresh=False)
                pbar.update(self.update_interval)
                self.logger.info("\n" + "-"*50)
            if self.config['debug'] and batch_idx >= 10:
                break

            self.global_step_count += 1
            if self.max_steps > 0 and self.global_step_count >= self.max_steps:
                if self.rank == 0:
                    self.logger.info(f"Reached max_steps={self.max_steps}, stopping training.")
                break

        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        torch.distributed.barrier()
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        torch.distributed.barrier()
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state()
        }

        self.lite.save(os.path.join(self.checkpoint_dir, self.saved_model_name), state=state)
        if self.rank == 0 and verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {self.saved_model_file}')

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learning_rate': self.config['learning_rate'],
            'weight_decay': self.config['weight_decay'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            k = k.replace('@', '_')
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})

    def to_device(self, data):
        device = self.device
        if isinstance(data, tuple) or isinstance(data, list):
            tdata = ()
            for d in data:
                d = d.to(device)
                tdata += (d,)
            return tdata
        elif isinstance(data, dict):
            for k, v in data.items():
                data[k] = v.to(device)
            return data
        else:
            return data.to(device)

    def _maybe_resume(self, saved=True):
        """Load model + optimizer + RNG state from the latest checkpoint in
        self.checkpoint_dir when config['auto_resume'] is truthy. Sets
        self.start_epoch and self._resume_batch_idx so the fit() loop and
        _train_epoch() pick up where the previous run stopped. No-op if
        auto_resume is off, the checkpoint dir is missing, or no checkpoint
        files are found (logs the reason and returns).

        Must be called AFTER self.lite.setup() and BEFORE torch.compile(),
        so state loads into the bare Fabric-wrapped model.
        """
        self._resume_batch_idx = 0  # default: no skip
        if not saved or not self.config.get('auto_resume', False):
            return
        if not os.path.isdir(self.checkpoint_dir):
            if self.rank == 0:
                self.logger.info(
                    f"auto_resume: no checkpoint dir at {self.checkpoint_dir}; starting fresh"
                )
            return

        import glob
        model_name = self.config['model']
        step_pat = os.path.join(self.checkpoint_dir, f'{model_name}-*-step-*.pth')
        epoch_pat = os.path.join(self.checkpoint_dir, f'{model_name}-*.pth')
        candidates = sorted(
            set(glob.glob(step_pat) + glob.glob(epoch_pat)),
            key=os.path.getmtime,
        )
        candidates = [c for c in candidates if os.path.exists(c)]
        if not candidates:
            if self.rank == 0:
                self.logger.info(
                    f"auto_resume: no checkpoint found in {self.checkpoint_dir}; starting fresh"
                )
            return

        latest = candidates[-1]
        is_step_ckpt = '-step-' in os.path.basename(latest)

        # DeepSpeed checkpoints are directories containing
        # checkpoint/mp_rank_00_model_states.pt (model + user-state scalars) and
        # checkpoint/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt (optimizer).
        # Fabric.load handles the model + optimizer; read scalars directly because
        # the Fabric API returns loaded state via in-place tensor update, not via
        # scalar return.
        inner = os.path.join(latest, 'checkpoint', 'mp_rank_00_model_states.pt')
        if not os.path.isfile(inner):
            if self.rank == 0:
                self.logger.warning(
                    f"auto_resume: {latest} has no inner state file; starting fresh"
                )
            return

        inner_state = torch.load(inner, map_location='cpu', weights_only=False)
        saved_epoch = int(inner_state.get('epoch', 0))
        saved_batch_idx = int(inner_state.get('batch_idx', 0)) if is_step_ckpt else 0
        saved_cur_step = int(inner_state.get('cur_step', 0))
        saved_best = inner_state.get('best_valid_score', self.best_valid_score)
        rng_state = inner_state.get('rng_state', None)
        cuda_rng_state = inner_state.get('cuda_rng_state', None)
        del inner_state

        # Load model + optimizer via Fabric (handles DeepSpeed ZeRO sharding).
        state = {"model": self.model, "optimizer": self.optimizer}
        self.lite.load(latest, state, strict=False)

        if rng_state is not None:
            if not isinstance(rng_state, torch.Tensor):
                rng_state = torch.tensor(rng_state, dtype=torch.uint8)
            torch.set_rng_state(rng_state)
        if cuda_rng_state is not None and torch.cuda.is_available():
            if not isinstance(cuda_rng_state, torch.Tensor):
                cuda_rng_state = torch.tensor(cuda_rng_state, dtype=torch.uint8)
            torch.cuda.set_rng_state(cuda_rng_state)

        if is_step_ckpt:
            self.start_epoch = saved_epoch
            self._resume_batch_idx = saved_batch_idx
        else:
            # End-of-epoch ckpt: continue from the NEXT epoch at batch 0.
            self.start_epoch = saved_epoch + 1
        self.cur_step = saved_cur_step
        self.best_valid_score = saved_best

        if self.rank == 0:
            kind = "step" if is_step_ckpt else "epoch"
            self.logger.info(
                f"auto_resume: Resuming from {os.path.basename(latest)} "
                f"({kind} ckpt: epoch={saved_epoch}, batch_idx={saved_batch_idx}, "
                f"cur_step={saved_cur_step}, best_valid_score={saved_best})"
            )

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.scheduler_config:
            warmup_rate = self.scheduler_config.get('warmup', 0.001)
            micro_steps = self.max_steps if self.max_steps > 0 else len(train_data) * self.epochs
            tot_steps = micro_steps // self.grad_accum_steps
            warmup_steps = tot_steps * warmup_rate
            self.lr_scheduler = self._build_scheduler(warmup_steps=warmup_steps, tot_steps=tot_steps)

        world_size, local_world_size = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
        nnodes = world_size // local_world_size
        precision = self.config['precision'] if self.config['precision'] else '32'
        if self.config['strategy'] == 'deepspeed':
            self.logger.info(f"Use deepspeed strategy")
            strategy = DeepSpeedStrategy(
                stage=self.config["stage"],
                precision=precision,
                exclude_frozen_parameters=self.config.get('exclude_frozen_parameters', True),
            )
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        else:
            self.logger.info(f"Use DDP strategy")
            strategy = DDPStrategy(find_unused_parameters=True)
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        self.lite.launch()
        self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)

        # Resume BEFORE torch.compile so model + optimizer state load into the
        # bare Fabric-wrapped model, not the compiled wrapper.
        self._maybe_resume(saved=saved)

        # Goal 4: torch.compile the Fabric-wrapped model (includes LoRA adapters)
        if self.config.get('torch_compile', False):
            compile_mode = self.config.get('torch_compile_mode', 'default')
            self.logger.info(
                f"torch.compile: enabled, mode='{compile_mode}' "
                f"(first step will stall ~60-120s for graph capture)"
            )
            self.model = torch.compile(self.model, mode=compile_mode)
        else:
            self.logger.info("torch.compile: disabled")

        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            if self.config['need_training'] == None or self.config['need_training']:
                train_data.sampler.set_epoch(epoch_idx)
                training_start_time = time()
                train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress, valid_data=valid_data)
                self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                if verbose:
                    self.logger.info(train_loss_output)
                if self.rank == 0:
                    self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
                self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx}, head='train')

            # Break out of epoch loop if max_steps reached
            if self.max_steps > 0 and self.global_step_count >= self.max_steps:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                break

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                    (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if self.rank == 0:
                    self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                    for name, value in valid_result.items():
                        self.tensorboard.add_scalar(name.replace('@', '_'), value, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                # Guard: suppress early-stop until at least `min_epochs_before_early_stop`
                # epochs have completed. With eval_step=1 + stopping_step=2, the default
                # upstream behaviour lets training exit after 3 epochs, which is noisy on
                # runs where loss hasn't even crossed log(N+1) random baseline yet.
                min_epochs_before_stop = self.config.get('min_epochs_before_early_stop', 1)
                if stop_flag and (epoch_idx + 1) < min_epochs_before_stop:
                    if verbose:
                        self.logger.info(
                            f"early-stop suppressed: epoch {epoch_idx + 1} < "
                            f"min_epochs_before_early_stop={min_epochs_before_stop}"
                        )
                    stop_flag = False

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                        (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def _full_sort_batch_eval(self, batched_data):
        user, time_seq, history_index, positive_u, positive_i = batched_data
        interaction = self.to_device(user)
        time_seq = self.to_device(time_seq)
        if self.config['model'] == 'HLLM':
            if self.config['stage'] == 3:
                scores = self.model.module.predict(interaction, time_seq, self.item_feature)
            else:
                scores = self.model((interaction, time_seq, self.item_feature), mode='predict')
        else:
            scores = self.model.module.predict(interaction, time_seq, self.item_feature)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return scores, positive_u, positive_i

    @torch.no_grad()
    def compute_item_feature(self, config, data):
        if self.use_text:
            item_data = BatchTextDataset(config, data)
            item_batch_size = config['MAX_ITEM_LIST_LENGTH'] * config['train_batch_size']
            item_loader = DataLoader(item_data, batch_size=item_batch_size, num_workers=14, shuffle=False, pin_memory=True, collate_fn=customize_rmpad_collate)
            self.logger.info(f"Inference item_data with {item_batch_size = } {len(item_loader) = }")
            self.item_feature = []
            with torch.no_grad():
                for idx, items in tqdm(enumerate(item_loader), total=len(item_loader)):
                    items = self.to_device(items)
                    items = self.model(items, mode='compute_item')
                    self.item_feature.append(items)
                if isinstance(items, tuple):
                    self.item_feature = torch.cat([x[0] for x in self.item_feature]), torch.cat([x[1] for x in self.item_feature])
                else:
                    self.item_feature = torch.cat(self.item_feature)
                if self.config['stage'] == 3:
                    self.item_feature = self.item_feature.bfloat16()
        else:
            with torch.no_grad():
                self.item_feature = self.model.module.compute_item_all()

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat.sum() / num_total_examples

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False, init_model=False):
        if not eval_data:
            return
        if init_model:
            world_size, local_world_size = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
            nnodes = world_size // local_world_size
            if self.config['strategy'] == 'deepspeed':
                self.logger.info(f"Use deepspeed strategy")
                precision = self.config['precision'] if self.config['precision'] else '32'
                strategy = DeepSpeedStrategy(
                    stage=self.config['stage'],
                    precision=precision,
                    exclude_frozen_parameters=self.config.get('exclude_frozen_parameters', True),
                )
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
                self.lite.launch()
                self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)
            else:
                self.logger.info(f"Use DDP strategy")
                precision = self.config['precision'] if self.config['precision'] else '32'
                strategy = DDPStrategy(find_unused_parameters=True)
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
                self.lite.launch()
                self.model = self.lite.setup(self.model)

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            state = {"model": self.model}
            self.lite.load(checkpoint_file, state, strict=False)
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        with torch.no_grad():
            self.model.eval()
            eval_func = self._full_sort_batch_eval

            self.tot_item_num = eval_data.dataset.dataload.item_num
            self.compute_item_feature(self.config, eval_data.dataset.dataload)
            iter_data = (
                tqdm(
                    eval_data,
                    total=len(eval_data),
                    ncols=150,
                    desc=set_color(f"Evaluate   ", 'pink'),
                    file=sys.stdout
                ) if show_progress and self.rank == 0 else eval_data
            )
            fwd_time = t.time()
            for batch_idx, batched_data in enumerate(iter_data):
                start_time = fwd_time
                data_time = t.time()
                scores, positive_u, positive_i = eval_func(batched_data)
                fwd_time = t.time()

                if show_progress and self.rank == 0:
                    iter_data.set_postfix_str(f"data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f}", refresh=False)
                self.eval_collector.eval_batch_collect(scores, positive_u, positive_i)
            num_total_examples = len(eval_data.sampler.dataset)
            struct = self.eval_collector.get_data_struct()
            result = self.evaluator.evaluate(struct)

            metric_decimal_place = 5 if self.config['metric_decimal_place'] == None else self.config['metric_decimal_place']
            for k, v in result.items():
                result_cpu = self.distributed_concat(torch.tensor([v]).to(self.device), num_total_examples).cpu()
                result[k] = round(result_cpu.item(), metric_decimal_place)
            self.wandblogger.log_eval_metrics(result, head='eval')

            return result
