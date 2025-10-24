# pl_trainer_generic.py
import os, json, random, gc
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
)

from peft import LoraConfig, get_peft_model

# ----------------------------
# Utilities (from your version)
# ----------------------------

def compute_plackett_luce_logprob(scores: torch.Tensor) -> torch.Tensor:
    """
    scores: [k] log-scores (or scores up to a monotone transform). Returns scalar log P(rank).
    """
    k = scores.shape[0]
    logp = scores.new_zeros(())
    for i in range(k - 1):
        rem = scores[i:]
        m = rem.max()
        logsumexp = m + torch.log(torch.exp(rem - m).sum())
        logp = logp + (scores[i] - logsumexp)
    return logp

def get_ranking_from_q_and_genorder(q_values, generation_order):
    if not isinstance(q_values, torch.Tensor):
        q_values = torch.tensor(q_values, dtype=torch.float32)
    if not isinstance(generation_order, torch.Tensor):
        generation_order = torch.tensor(generation_order, dtype=torch.float32)
    epsilon = 1e-6
    composite = q_values - epsilon * generation_order
    return torch.argsort(composite, descending=True).tolist()

# ---------------------------------------
# Data preparation (tokenize per-candidate)
# ---------------------------------------

def prepare_dataset_for_training(
    data_list: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
) -> List[Dict]:
    processed = []
    for sample in data_list:
        messages = sample["messages"]
        candidates = sample["candidates"]
        q_values = sample["q_values"]
        generation_order = sample.get("generation_order", list(range(len(candidates))))

        input_ids_list, attn_list, labels_list = [], [], []

        for cand in candidates:
            full_conv = messages + [cand]

            # conversation with candidate
            full_text = tokenizer.apply_chat_template(
                full_conv, tokenize=False, add_generation_prompt=False
            )
            # prompt only (mask these tokens in labels)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            full_enc = tokenizer(full_text, add_special_tokens=True, max_length=max_length, truncation=True)
            prompt_enc = tokenizer(prompt_text, add_special_tokens=True, max_length=max_length, truncation=True)

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]

            labels = input_ids.copy()
            prompt_len = len(prompt_enc["input_ids"])
            labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

            input_ids_list.append(input_ids)
            attn_list.append(attention_mask)
            labels_list.append(labels)

        processed.append(
            {
                "input_ids": input_ids_list,          # list[list[int]] (K x T_i)
                "attention_mask": attn_list,          # list[list[int]]
                "labels": labels_list,                # list[list[int]]
                "q_values": q_values,                 # list[float]
                "generation_order": generation_order, # list[int]
                "num_candidates": len(candidates),
            }
        )
    return processed

class ListwiseRankDataset(TorchDataset):
    def __init__(self, records: List[Dict]):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx): return self.records[idx]

# --------------------------------------
# Collator: flatten [B, K, T] → [∑K, T]
# --------------------------------------

class ListwiseCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, label_pad_token_id=-100, pad_token_id=None):
        self.tokenizer = tokenizer
        self.label_pad = label_pad_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # counts per example (K_i) and rankings to compute loss later
        counts = [len(f["input_ids"]) for f in features]

        # compute final ranking indices per example using q_values + generation_order
        rankings = []
        for f in features:
            rankings.append(
                torch.tensor(
                    get_ranking_from_q_and_genorder(f["q_values"], f["generation_order"]),
                    dtype=torch.long,
                )
            )

        # find max length across all candidates in batch
        max_len = 0
        for f in features:
            for seq in f["input_ids"]:
                if len(seq) > max_len: max_len = len(seq)

        def pad_list(seq, pad_id): return seq + [pad_id] * (max_len - len(seq))

        flat_input_ids, flat_attention, flat_labels = [], [], []
        for f in features:
            for ids, attn, lab in zip(f["input_ids"], f["attention_mask"], f["labels"]):
                flat_input_ids.append(pad_list(ids, self.pad_token_id))
                flat_attention.append(pad_list(attn, 0))
                flat_labels.append(pad_list(lab, self.label_pad))

        batch = {
            "input_ids": torch.tensor(flat_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(flat_attention, dtype=torch.long),
            "labels": torch.tensor(flat_labels, dtype=torch.long),
            "counts": torch.tensor(counts, dtype=torch.long),  # shape [B]
            # pack rankings into a single tensor with padding to K_max
        }

        K_max = max(counts)
        rank_pad = -1
        packed_rankings = []
        for r in rankings:
            if r.numel() < K_max:
                r = torch.cat([r, torch.full((K_max - r.numel(),), rank_pad, dtype=torch.long)], dim=0)
            packed_rankings.append(r)
        batch["rankings"] = torch.stack(packed_rankings, dim=0)  # [B, K_max], padded with -1

        return batch

# ------------------------------------------------
# Generic Trainer subclass with PL-DPO compute_loss
# ------------------------------------------------

class PLGenericTrainer(Trainer):
    def __init__(self, *args, ref_model: Optional[torch.nn.Module] = None, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert ref_model is not None, "ref_model is required"
        self.ref_model = ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.beta = beta
        self.label_pad_token_id = -100

    @torch.no_grad()
    def _seq_logprobs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, T, V]
        labels: [N, T]
        returns: [N] sum of log p(label_t | < t) over non-pad labels
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_mask = shift_labels != self.label_pad_token_id
        logp_tok = F.log_softmax(shift_logits, dim=-1)
        gather_index = shift_labels.masked_fill(~loss_mask, 0).unsqueeze(-1)
        logp = torch.gather(logp_tok, 2, gather_index).squeeze(-1)
        logp = (logp * loss_mask.float()).sum(dim=1)  # [N]
        return logp

    def compute_loss(self, model, inputs, return_outputs=False):
        # Flattened batch
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        counts = inputs["counts"]          # [B]
        rankings = inputs["rankings"]      # [B, K_max] (padded with -1)

        # Forward through policy
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pol_seq_logp = self._seq_logprobs(outputs.logits, labels)   # [sum_k]

        # Forward through reference (no grad)
        with torch.no_grad():
            ref_out = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_seq_logp = self._seq_logprobs(ref_out.logits, labels)  # [sum_k]

        # Split back per-example
        split_pol = list(torch.split(pol_seq_logp, counts.tolist()))
        split_ref = list(torch.split(ref_seq_logp, counts.tolist()))
        B = len(split_pol)
        K_max = rankings.size(1)

        # Left-pad with -inf to K_max so we can index safely
        ninf = torch.finfo(pol_seq_logp.dtype).min
        pol_mat, ref_mat = [], []
        for p, r in zip(split_pol, split_ref):
            if p.numel() < K_max:
                pad = p.new_full((K_max - p.numel(),), ninf)
                p = torch.cat([p, pad], dim=0)
                r = torch.cat([r, pad], dim=0)
            pol_mat.append(p)
            ref_mat.append(r)
        pol_mat = torch.stack(pol_mat, dim=0)  # [B, K_max]
        ref_mat = torch.stack(ref_mat, dim=0)  # [B, K_max]

        # For each example: reorder by ranking (ignore padded -1)
        total_loss = 0.0
        # For logging (first item only)
        metrics = {}

        for i in range(B):
            rank = rankings[i]
            valid_mask = rank != -1
            idx = rank[valid_mask]                # [K_i]
            pol_ranked = pol_mat[i].index_select(0, idx)
            ref_ranked = ref_mat[i].index_select(0, idx)

            # PL log-probs for policy and reference
            logP_pol = compute_plackett_luce_logprob(pol_ranked)
            logP_ref = compute_plackett_luce_logprob(ref_ranked)

            # DPO-style contrastive objective at list level
            margin = self.beta * (logP_pol - logP_ref)
            loss_i = -F.logsigmoid(margin)
            total_loss = total_loss + loss_i

            if i == 0 and self.state.is_world_process_zero:
                metrics["train/logP_pol_list"] = logP_pol.detach().float().item()
                metrics["train/logP_ref_list"] = logP_ref.detach().float().item()
                metrics["train/logP_margin"] = (logP_pol - logP_ref).detach().float().item()
                metrics["train/reward_margin_beta"] = margin.detach().float().item()

        loss = total_loss / B

        # push metrics (avoids flooding)
        if metrics:
            self.log(metrics)

        return (loss, outputs) if return_outputs else loss

# -------------------
# Training orchestration
# -------------------

def load_and_split_jsonl(path: str, split_ratio: float = 0.9, seed: int = 42):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    random.Random(seed).shuffle(data)
    n_train = int(len(data) * split_ratio)
    return data[:n_train], data[n_train:]

def train_plackett_luce_generic(
    model_name: str,
    data_path: str,
    output_dir: str = "./pl_generic_output",
    num_epochs: int = 3,
    max_steps: Optional[int] = None,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    eval_steps: int = 500,
    save_strategy: str = "epoch",
    logging_steps: int = 10,
    warmup_steps: int = 100,
    max_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
):
    # tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # models
    policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()
    for p in ref.parameters(): p.requires_grad_(False)

    # LoRA on policy (optional)
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    policy = get_peft_model(policy, lora_cfg)
    if gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.print_trainable_parameters()

    # data
    train_raw, eval_raw = load_and_split_jsonl(data_path, split_ratio=0.9, seed=42)
    train_records = prepare_dataset_for_training(train_raw, tok, max_length=max_length)
    eval_records  = prepare_dataset_for_training(eval_raw,  tok, max_length=max_length)
    train_ds = ListwiseRankDataset(train_records)
    eval_ds  = ListwiseRankDataset(eval_records)

    # collator
    collator = ListwiseCollator(tok, label_pad_token_id=-100, pad_token_id=tok.pad_token_id)

    # args
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps else -1,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        evaluation_strategy="steps" if len(eval_ds) > 0 else "no",
        eval_steps=eval_steps if len(eval_ds) > 0 else None,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        report_to="none",
        bf16=True,
        remove_unused_columns=False,  # important: we pass custom tensors
        save_total_limit=3,
        load_best_model_at_end=False,
    )

    trainer = PLGenericTrainer(
        model=policy,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) > 0 else None,
        tokenizer=tok,
        data_collator=collator,
        ref_model=ref,
        beta=beta,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))
    return policy, tok

# if __name__ == "__main__":
#     policy, tok = train_plackett_luce_generic(
#         model_name="Qwen/Qwen2.5-0.5B-Instruct",
#         data_path="path/to/your.jsonl",
#         output_dir="./pl_dpo_generic",
#     )
