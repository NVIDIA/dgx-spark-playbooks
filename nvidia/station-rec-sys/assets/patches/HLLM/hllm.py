# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from logging import getLogger

from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel, all_gather
from REC.model.HLLM.modeling_llama import LlamaForCausalLM

import torch._dynamo
# Treat nn.Module int attrs (e.g. layer_idx) as dynamic so 40 transformer layers share one compiled graph instead of recompiling per layer.
torch._dynamo.config.allow_unspec_int_on_nn_module = True
from REC.model.HLLM.modeling_mistral import MistralForCausalLM
from REC.model.HLLM.modeling_bert import BertModel
from REC.model.HLLM.baichuan.modeling_baichuan import BaichuanForCausalLM

try:
    from peft import LoraConfig, inject_adapter_in_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class HLLM(BaseModel):
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(HLLM, self).__init__()
        self.logger = getLogger()

        self.item_pretrain_dir = config['item_pretrain_dir']
        self.user_pretrain_dir = config['user_pretrain_dir']
        self.gradient_checkpointing = config['gradient_checkpointing']
        self.use_ft_flash_attn = config['use_ft_flash_attn']
        self.logger.info(f"create item llm")
        self.item_llm = self.create_llm(self.item_pretrain_dir, config['item_llm_init'])
        self.logger.info(f"create user llm")
        self.user_llm = self.create_llm(self.user_pretrain_dir, config['user_llm_init'])

        # Apply LoRA if configured
        lora_r = config.get('lora_r', 0)
        if lora_r > 0:
            if not HAS_PEFT:
                raise ImportError("peft is required for LoRA. Install with: pip install peft")
            lora_alpha = config.get('lora_alpha', lora_r * 4)
            lora_dropout = config.get('lora_dropout', 0.05)
            lora_targets = config.get('lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_targets,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.logger.info(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}, "
                           f"dropout={lora_dropout}, targets={lora_targets}")
            inject_adapter_in_model(lora_config, self.item_llm, adapter_name="default")
            inject_adapter_in_model(lora_config, self.user_llm, adapter_name="default")
            # Count trainable params
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            self.logger.info(f"LoRA: {trainable:,} trainable / {total:,} total params "
                           f"({trainable/total*100:.2f}%)")

        self.item_emb_token_n = config['item_emb_token_n']
        if self.item_emb_token_n > 1:
            raise NotImplementedError(f"Not support item_emb_token_n {self.item_emb_token_n} > 1")

        if self.item_emb_token_n > 0:
            self.item_emb_tokens = nn.Parameter(
                torch.zeros(1, self.item_emb_token_n, self.item_llm.config.hidden_size)
            )
            self.item_emb_tokens.data.normal_(mean=0.0, std=0.02)
            if config['item_emb_pretrain']:
                ckpt = torch.load(config['item_emb_pretrain'], map_location='cpu')
                self.logger.info(f"load item_emb_token from {config['item_emb_pretrain']} with {ckpt.size()}")
                self.item_emb_tokens.data = nn.Parameter(ckpt)
        else:  # mean pooling
            self.item_emb_tokens = None

        self.loss = config['loss']
        if self.loss == 'nce':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.nce_thres = config['nce_thres'] if config['nce_thres'] else 0.99
            self.num_negatives = config['num_negatives']
            self.logger.info(f"nce thres setting to {self.nce_thres}")
        else:
            raise NotImplementedError(f"Only nce is supported")

        if config['load_pretrain']:
            state_dict = torch.load(config['load_pretrain'], map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            self.logger.info(f"{msg.missing_keys = }")
            self.logger.info(f"{msg.unexpected_keys = }")

    def create_llm(self, pretrain_dir, init=True):
        self.logger.info(f"******* create LLM {pretrain_dir} *******")
        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        self.logger.info(f"hf_config: {hf_config}")
        hf_config.gradient_checkpointing = self.gradient_checkpointing
        hf_config.use_cache = False
        hf_config.output_hidden_states = True
        hf_config.return_dict = True

        self.logger.info("xxxxx starting loading checkpoint")
        if isinstance(hf_config, transformers.LlamaConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for llama')
            self.logger.info(f'Init {init} for llama')
            if init:
                return LlamaForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return LlamaForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.MistralConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for mistral')
            self.logger.info(f'Init {init} for mistral')
            if init:
                return MistralForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return MistralForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.BertConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for bert')
            self.logger.info(f'Init {init} for bert')
            if init:
                return BertModel.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BertModel(config=hf_config).cuda()
        elif getattr(hf_config, "model_type", None) == "baichuan":
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for baichuan')
            self.logger.info(f'Init {init} for baichuan')
            if init:
                return BaichuanForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BaichuanForCausalLM(config=hf_config).cuda()
        else:
            self.logger.info(f'Using AutoModel fallback for {getattr(hf_config, "model_type", "unknown")}')
            self.logger.info(f'Init {init}')
            if init:
                model = AutoModelForCausalLM.from_pretrained(
                    pretrain_dir,
                    config=hf_config,
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                )
            else:
                model = AutoModelForCausalLM.from_config(config=hf_config).cuda()
            # HLLM reads only hidden_states; skip LM head to avoid materializing [tokens, vocab] logits (~8 GB for Qwen3).
            if hasattr(model, "lm_head") and not isinstance(model.lm_head, nn.Identity):
                model.lm_head = nn.Identity()
                self.logger.info(f"Replaced lm_head with Identity (skip logit materialization)")
            self.logger.info(
                f"Loaded {pretrain_dir} — attn_implementation="
                f"{model.config._attn_implementation}"
            )
            return model

    def nce_loss(self, cur_embs, target_pos, target_neg, user_attention_mask):
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        logit_scale = self.logit_scale.exp()
        D = target_neg.size(-1)
        output_embs = cur_embs / cur_embs.norm(dim=-1, keepdim=True)
        target_pos_embs = target_pos / target_pos.norm(dim=-1, keepdim=True)
        pos_logits = F.cosine_similarity(output_embs, target_pos_embs, dim=-1).unsqueeze(-1)

        target_neg = target_neg / target_neg.norm(dim=-1, keepdim=True)

        neg_embedding_all = all_gather(target_neg, sync_grads=True).reshape(-1, D)  # [num, dim]
        neg_embedding_all = neg_embedding_all.transpose(-1, -2)
        neg_logits = torch.matmul(output_embs, neg_embedding_all)
        fix_logits = torch.matmul(target_pos_embs, neg_embedding_all)
        neg_logits[fix_logits > self.nce_thres] = torch.finfo(neg_logits.dtype).min

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[user_attention_mask.bool()] * logit_scale
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        return logits, labels

    def forward_item_emb(
        self,
        input_ids,
        position_ids,
        cu_input_lens,
        emb_token_n,
        emb_tokens,
        llm
    ):
        inputs_embeds = llm.get_input_embeddings()(input_ids).clone()
        emb_pos = cu_input_lens.cumsum(dim=0, dtype=torch.int32)
        if emb_token_n > 0:
            inputs_embeds[emb_pos - 1] = emb_tokens
        model_out = llm(inputs_embeds=inputs_embeds.unsqueeze(0), cu_input_lens=cu_input_lens, position_ids=position_ids.unsqueeze(0))
        model_out = model_out.hidden_states[-1].squeeze(0)

        if emb_token_n > 0:
            emb = model_out[emb_pos - 1]
        else:
            max_len = cu_input_lens.max().item()
            cu_seqlens = F.pad(cu_input_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
            seqs = [model_out[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
            padded_seqs = [
                F.pad(
                    seqs[i],
                    (0, 0) * (seqs[i].dim() - 1) + (0, max_len - cu_input_lens[i]),
                    value=0.0,
                )
                for i in range(cu_input_lens.size(0))
            ]
            out = torch.stack(padded_seqs)
            emb = out.sum(dim=1) / cu_input_lens.unsqueeze(1)

        return emb

    def forward(self, interaction, mode='train'):
        if mode == 'predict':
            return self.predict(interaction[0], interaction[1], interaction[2])
        if mode == 'compute_item':
            return self.compute_item(interaction)
        user_attention_mask = interaction['attention_mask']
        N, S = user_attention_mask.shape
        pos_input_ids, pos_cu_input_lens, pos_position_ids = interaction['pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids']
        neg_input_ids, neg_cu_input_lens, neg_position_ids = interaction['neg_input_ids'], interaction['neg_cu_input_lens'], interaction['neg_position_ids']

        pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        pos_embedding = pos_embedding.reshape(N, S+1, -1)
        neg_embedding = self.forward_item_emb(neg_input_ids, neg_position_ids, neg_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        neg_embedding = neg_embedding.reshape(N, -1, self.item_llm.config.hidden_size)

        target_pos_embs = pos_embedding[:, 1:]
        target_neg_embs = neg_embedding

        user_embedding = self.user_llm(inputs_embeds=pos_embedding[:, :-1], attention_mask=user_attention_mask).hidden_states[-1]

        model_out = {}
        logits, labels = self.nce_loss(user_embedding, target_pos_embs, target_neg_embs, user_attention_mask)
        model_out['loss'] = F.cross_entropy(logits, labels)
        model_out['nce_samples'] = (logits > torch.finfo(logits.dtype).min/100).sum(dim=1).float().mean()  # samples after filtering same negatives
        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(1):
                break
            indices = logits.topk(k, dim=1).indices
            model_out[f"nce_top{k}_acc"] = labels.view(-1, 1).eq(indices).any(dim=1).float().mean()
        return model_out

    @torch._dynamo.disable
    @torch.no_grad()
    def predict(self, item_seq, time_seq, item_feature):
        attention_mask = (item_seq > 0).int()

        pos_embedding = item_feature[item_seq]

        user_embedding = self.user_llm(inputs_embeds=pos_embedding, attention_mask=attention_mask).hidden_states[-1]
        seq_output = user_embedding[:, -1]
        seq_output = seq_output / seq_output.norm(dim=-1, keepdim=True)
        item_feature = item_feature / item_feature.norm(dim=-1, keepdim=True)

        return torch.matmul(seq_output, item_feature.t())

    @torch.no_grad()
    def compute_item_all(self):
        return self.item_embedding.weight

    @torch.no_grad()
    def compute_item(self, interaction):
        pos_input_ids, pos_cu_input_lens, pos_position_ids = interaction['pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids']
        pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        N = pos_cu_input_lens.size(0)
        pos_embedding = pos_embedding.view(N, -1)

        return pos_embedding
