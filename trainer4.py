# pl_trainer.py
import os
import gc
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -------------------- Core utilities --------------------

def plackett_luce_loglik(scores: torch.Tensor) -> torch.Tensor:
    """
    scores: [k] tensor ordered best -> worst
    returns: scalar log-likelihood under PL
    """
    k = scores.shape[0]
    log_prob = scores.new_zeros(())
    for i in range(k - 1):
        remaining_scores = scores[i:]
        max_score = remaining_scores.max()
        logsumexp = max_score + torch.log(torch.exp(remaining_scores - max_score).sum())
        log_prob = log_prob + scores[i] - logsumexp
    return log_prob


def get_sequence_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_idx: int,
    requires_grad: bool = True,
) -> torch.Tensor:
    """
    Length-normalized log-prob over assistant response tokens only.
    input_ids / attention_mask: [1, seq_len]
    """
    with torch.set_grad_enabled(requires_grad):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab]

        # next-token prediction shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, dim=2, index=shift_labels.unsqueeze(2)
        ).squeeze(2)  # [1, seq_len-1]

        # in shifted space, position i-1 predicts token i
        resp_mask = torch.zeros_like(per_token_logps)
        if response_start_idx > 0:
            resp_mask[:, response_start_idx - 1:] = 1.0

        masked_logps = per_token_logps * resp_mask
        seq_logprob = masked_logps.sum()
        n_resp = resp_mask.sum().clamp(min=1.0)
        return seq_logprob / n_resp


def compute_candidate_logprobs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    sample: Dict[str, Any],
    device: torch.device,
    requires_grad: bool = True,
) -> torch.Tensor:
    """
    Returns [k] tensor of log-probs for each candidate in sample.
    """
    messages = sample["messages"]
    candidates = sample["candidates"]
    logprobs: List[torch.Tensor] = []

    # template without candidate to locate response start
    text_wo = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc_wo = tokenizer(text_wo, return_tensors="pt")
    response_start_idx = enc_wo["input_ids"].shape[1]

    for cand in candidates:
        full_conversation = messages + [cand]
        text = tokenizer.apply_chat_template(
            full_conversation, tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        lp = get_sequence_logprob(
            model, input_ids, attention_mask, response_start_idx, requires_grad=requires_grad
        )
        logprobs.append(lp)

    return torch.stack(logprobs)  # [k]


# -------------------- Data plumbing --------------------

class JsonlListDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if len(self.rows) == 0:
            raise ValueError(f"No records found in {path}")

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]


@dataclass
class PassThroughCollator:
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    def __call__(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # No tensorization; compute_loss handles everything per-sample.
        return examples


# -------------------- Trainer with PL + optional reference --------------------

class PLTrainer(Trainer):
    def __init__(
        self,
        ref_model: Optional[torch.nn.Module],
        tokenizer: AutoTokenizer,
        beta: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ref_model = ref_model.eval() if ref_model is not None else None
        if self.ref_model is not None:
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.device
        batch = inputs  # list[dict]
        losses = []

        for sample in batch:
            # 1) policy log-probs (requires grad)
            logps = compute_candidate_logprobs(
                model, self.tokenizer, sample, device, requires_grad=True
            )  # [k]

            # 2) optional reference scores
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_logps = compute_candidate_logprobs(
                        self.ref_model, self.tokenizer, sample, device, requires_grad=False
                    )
                scores = self.beta * (logps - ref_logps)
            else:
                scores = logps

            # 3) order best->worst
            if "generation_order" in sample and sample["generation_order"]:
                order = torch.tensor(sample["generation_order"], device=scores.device, dtype=torch.long)
                scores_ordered = scores[order]
            else:
                scores_ordered = scores  # assume already best->worst

            # 4) PL NLL
            ll = plackett_luce_loglik(scores_ordered)
            losses.append(-ll)

        loss = torch.stack(losses).mean()
        return loss  # we don't need outputs; keep it minimal

    @torch.no_grad()
    def evaluate_pl_metrics(self, dataset) -> Dict[str, float]:
        """
        Mean PL log-likelihood and top-1 accuracy on eval set.
        """
        device = self.model.device
        dl = self.get_eval_dataloader(dataset)
        total_ll, total_n = 0.0, 0
        total_top1, total_top1_n = 0, 0

        for batch in dl:
            for sample in batch:
                logps = compute_candidate_logprobs(
                    self.model, self.tokenizer, sample, device, requires_grad=False
                )
                if self.ref_model is not None:
                    ref_logps = compute_candidate_logprobs(
                        self.ref_model, self.tokenizer, sample, device, requires_grad=False
                    )
                    scores = self.beta * (logps - ref_logps)
                else:
                    scores = logps

                if "generation_order" in sample and sample["generation_order"]:
                    order = torch.tensor(sample["generation_order"], device=scores.device, dtype=torch.long)
                    scores_ordered = scores[order]
                    best_truth = order[0].item()
                else:
                    scores_ordered = scores
                    best_truth = 0

                ll = plackett_luce_loglik(scores_ordered)
                total_ll += ll.item()
                total_n += 1

                pred_idx = int(torch.argmax(scores).item())
                total_top1 += int(pred_idx == best_truth)
                total_top1_n += 1

        mean_ll = (total_ll / total_n) if total_n else float("nan")
        top1 = (total_top1 / total_top1_n) if total_top1_n else float("nan")
        return {"eval_pl_loglik": mean_ll, "eval_top1_acc": top1}


# -------------------- Callbacks --------------------

class MemoryCleanupCallback(TrainerCallback):
    def __init__(self, every_n_steps: int = 3):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        return control


class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[step {state.global_step}] loss={logs['loss']:.4f}")
        return control


class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            self.tokenizer.save_pretrained(ckpt_dir)
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.tokenizer.save_pretrained(args.output_dir)
        return control


class EvalMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        trainer: Optional[PLTrainer] = kwargs.get("trainer", None)
        if trainer is None or trainer.eval_dataset is None:
            return control
        metrics = trainer.evaluate_pl_metrics(trainer.eval_dataset)
        # log to console
        print(f"[eval step {state.global_step}] {metrics}")
        # also feed into Trainer's metric store (still console only with report_to=[])
        trainer.log(metrics)
        return control


# -------------------- Main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, default=None)

    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--ref_model_name", type=str, default=None,
                   help="Optional reference policy; if set, use scores = beta*(logpi - logpi_ref)")
    p.add_argument("--beta", type=float, default=1.0)

    p.add_argument("--output_dir", type=str, default="./pl_out")
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--evaluation_steps", type=int, default=200)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    p.add_argument("--memory_cleanup_every", type=int, default=3)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated module names to adapt.")

    args = p.parse_args()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # trainable model (LoRA)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = prepare_model_for_kbit_training(model)  # safe if not using 4/8-bit
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    # optional reference model (frozen; no LoRA)
    ref_model = None
    if args.ref_model_name:
        ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_name)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # datasets + collator
    train_ds = JsonlListDataset(args.train_jsonl)
    eval_ds = JsonlListDataset(args.eval_jsonl) if args.eval_jsonl else None
    collator = PassThroughCollator(tokenizer=None)

    # training args (no external logging backends)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_pin_memory=False,
        report_to=[],  # don't hook anywhere
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.evaluation_steps if eval_ds is not None else None,
    )

    # trainer
    trainer = PLTrainer(
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=args.beta,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # callbacks
    trainer.add_callback(MemoryCleanupCallback(every_n_steps=args.memory_cleanup_every))
    trainer.add_callback(LossLoggingCallback())
    trainer.add_callback(SaveTokenizerCallback(tokenizer))
    if eval_ds is not None:
        trainer.add_callback(EvalMetricsCallback())

    # train
    train_result = trainer.train()
    # save adapter + tokenizer + trainer state
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    # (optional) final eval if eval set given
    if eval_ds is not None:
        metrics = trainer.evaluate()
        # also compute PL metrics explicitly for clarity
        pl_metrics = trainer.evaluate_pl_metrics(eval_ds)
        print("[final eval] hf_metrics:", metrics)
        print("[final eval] pl_metrics:", pl_metrics)


if __name__ == "__main__":
    main()
