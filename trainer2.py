import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import gc

def pl_data_collator(features: List[Dict]) -> Dict[str, List]:
    """
    Keep examples as python lists (ragged). We do padding/tensorization later.
    """
    out = {}
    keys = features[0].keys()
    for k in keys:
        out[k] = [f[k] for f in features]
    return out


def compute_plackett_luce_logprob(scores: torch.Tensor):
    """
    scores: [k] tensor (best → worst), returns scalar tensor (log-prob)
    """
    k = scores.shape[0]
    log_prob = scores.new_zeros(())  # tensor scalar on same device/dtype
    for i in range(k - 1):
        remaining_scores = scores[i:]
        max_score = remaining_scores.max()
        logsumexp = max_score + torch.log(torch.exp(remaining_scores - max_score).sum())
        log_prob = log_prob + (scores[i] - logsumexp)
    return log_prob



def get_ranking_from_q_and_genorder(q_values, generation_order):
    """
    Create ranking: primary by Q-values, secondary by generation order for ties.

    Args:
        q_values: list or tensor - Q values from MCTS
        generation_order: list or tensor - generation order (0, 1, 2, ...)

    Returns:
        ranking_indices: list - indices in ranked order (best to worst)
    """
    if not isinstance(q_values, torch.Tensor):
        q_values = torch.tensor(q_values, dtype=torch.float32)
    if not isinstance(generation_order, torch.Tensor):
        generation_order = torch.tensor(generation_order, dtype=torch.float32)

    # Create composite score
    epsilon = 1e-6
    composite_scores = q_values - epsilon * generation_order

    # Sort descending (best first)
    ranking_indices = torch.argsort(composite_scores, descending=True)

    return ranking_indices.tolist()


class PlackettLuceDPOTrainer(DPOTrainer):
    """
    Custom DPO Trainer for Plackett-Luce ranking with k > 2 candidates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track global step for logging
        self.global_step = 0

    def get_batch_samples(self, model, batch: Dict[str, torch.Tensor]) -> Tuple[str, str]:
        """Override to handle k candidates instead of just chosen/rejected."""
        # This method is used by parent for logging, we'll handle differently
        return "", ""

    def concatenated_inputs(
            self,
            batch: Dict[str, Union[List, torch.LongTensor]],
            is_encoder_decoder: bool = False,
            label_pad_token_id: int = -100,
            padding_value: int = 0,
            device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """
        Concatenate inputs for all k candidates.
        Modified from DPOTrainer to handle k > 2.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            raise NotImplementedError("Encoder-decoder models not supported yet")

        # Handle k candidates
        max_length = 0

        # Find max length across all candidates
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch:
                # batch[key] is a list of lists (one per candidate per sample)
                for candidates in batch[key]:
                    for candidate in candidates:
                        max_length = max(max_length, len(candidate))

        # Pad and concatenate
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch:
                pad_value = padding_value if key != "labels" else label_pad_token_id

                # Flatten: [batch_size, k, seq_len] -> [batch_size * k, seq_len]
                all_candidates = []
                for candidates in batch[key]:
                    for candidate in candidates:
                        # Pad to max_length
                        padded = candidate + [pad_value] * (max_length - len(candidate))
                        all_candidates.append(padded)

                concatenated_batch[key] = torch.tensor(all_candidates, device=device)

        return concatenated_batch

    def concatenated_forward(
            self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Forward all candidates, return [B, K_max] tensors; padded slots are -inf.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )

        # counts of candidates per sample (variable k)
        counts = [len(cands) for cands in batch["input_ids"]]
        B = len(counts)
        K_max = max(counts)

        # policy forward
        outputs = model(
            input_ids=concatenated_batch["input_ids"],
            attention_mask=concatenated_batch["attention_mask"],
        )
        logits = outputs.logits
        labels = concatenated_batch["labels"]

        # shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)

        # mask BEFORE gather; set dummy index for masked positions
        loss_mask = (shift_labels != self.label_pad_token_id)
        gather_index = shift_labels.clone()
        gather_index[~loss_mask] = 0  # safe dummy index

        per_token_logps = torch.gather(log_probs, dim=2, index=gather_index.unsqueeze(2)).squeeze(2)
        per_token_logps = per_token_logps * loss_mask.float()
        sequence_logps = per_token_logps.sum(dim=1)  # [sum_k]

        # split back per sample and left-pad with -inf to K_max
        split_policy = list(torch.split(sequence_logps, counts))
        pad_fill = torch.finfo(sequence_logps.dtype).min  # ~ -inf
        policy_list = []
        for s in split_policy:
            if s.numel() < K_max:
                pad = s.new_full((K_max - s.numel(),), pad_fill)
                s = torch.cat([s, pad], dim=0)
            policy_list.append(s)
        policy_logprobs = torch.stack(policy_list, dim=0)  # [B, K_max]

        # reference forward (eval/no-grad)
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=concatenated_batch["input_ids"],
                attention_mask=concatenated_batch["attention_mask"],
            )
            ref_logits = ref_outputs.logits
            ref_shift_logits = ref_logits[:, :-1, :].contiguous()
            ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)

            ref_per_token_logps = torch.gather(ref_log_probs, dim=2, index=gather_index.unsqueeze(2)).squeeze(2)
            ref_per_token_logps = ref_per_token_logps * loss_mask.float()
            ref_sequence_logps = ref_per_token_logps.sum(dim=1)

            split_ref = list(torch.split(ref_sequence_logps, counts))
            ref_list = []
            for s in split_ref:
                if s.numel() < K_max:
                    pad = s.new_full((K_max - s.numel(),), pad_fill)
                    s = torch.cat([s, pad], dim=0)
                ref_list.append(s)
            reference_logprobs = torch.stack(ref_list, dim=0)  # [B, K_max]

        return policy_logprobs, reference_logprobs, None, None

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: str = "train",
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Plackett-Luce DPO loss and metrics.
        """
        metrics = {}

        # Get log probs for all candidates
        policy_logprobs, reference_logprobs, _, _ = self.concatenated_forward(model, batch)

        batch_size = policy_logprobs.shape[0]
        losses = []

        # Compute loss for each sample in batch
        for i in range(batch_size):
            # Get ranking for this sample
            q_values = batch["q_values"][i]
            generation_order = batch.get("generation_order", [list(range(len(q_values)))])[i]

            ranking_indices = get_ranking_from_q_and_genorder(q_values, generation_order)
            ranking_indices_tensor = torch.tensor(ranking_indices, device=policy_logprobs.device)

            # Reorder logprobs according to ranking
            policy_scores_ranked = policy_logprobs[i][ranking_indices_tensor]
            ref_scores_ranked = reference_logprobs[i][ranking_indices_tensor]

            # Compute Plackett-Luce log probabilities
            log_p_policy = compute_plackett_luce_logprob(policy_scores_ranked)
            log_p_ref = compute_plackett_luce_logprob(ref_scores_ranked)

            # DPO loss: -log sigmoid(beta * (log_p_policy - log_p_ref))
            logits = self.beta * (log_p_policy - log_p_ref)
            loss = -F.logsigmoid(logits)

            losses.append(loss)

            # Accumulate metrics
            if i == 0:  # Log metrics for first sample in batch
                metrics[f"{train_eval}/logprobs_policy_ranking"] = log_p_policy.item()
                metrics[f"{train_eval}/logprobs_ref_ranking"] = log_p_ref.item()
                metrics[f"{train_eval}/logprobs_margin"] = (log_p_policy - log_p_ref).item()
                metrics[f"{train_eval}/rewards_margin"] = logits.item()
                metrics[f"{train_eval}/rewards_accuracy"] = (logits > 0).float().item()

                # Best and worst candidate log probs
                metrics[f"{train_eval}/logprobs_policy_best"] = policy_logprobs[i][ranking_indices_tensor[0]].item()
                metrics[f"{train_eval}/logprobs_policy_worst"] = policy_logprobs[i][ranking_indices_tensor[-1]].item()
                metrics[f"{train_eval}/logprobs_ref_best"] = reference_logprobs[i][ranking_indices_tensor[0]].item()
                metrics[f"{train_eval}/logprobs_ref_worst"] = reference_logprobs[i][ranking_indices_tensor[-1]].item()

        # Average loss across batch
        loss = torch.stack(losses).mean()
        metrics[f"{train_eval}/loss"] = loss.item()

        return loss, metrics


def prepare_dataset_for_training(data_list: List[Dict], tokenizer) -> Dataset:
    """
    Prepare dataset in the format expected by PlackettLuceDPOTrainer.

    Args:
        data_list: List of dicts with "messages", "candidates", "q_values", "generation_order"
        tokenizer: HuggingFace tokenizer

    Returns:
        HuggingFace Dataset object
    """
    processed_data = []

    for sample in data_list:
        messages = sample["messages"]
        candidates = sample["candidates"]
        q_values = sample["q_values"]
        generation_order = sample.get("generation_order", list(range(len(candidates))))

        # Tokenize each candidate
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for candidate in candidates:
            full_conversation = messages + [candidate]

            # Apply chat template
            text = tokenizer.apply_chat_template(
                full_conversation,
                tokenize=False,
                add_generation_prompt=False
            )

            # Get prompt text (without candidate)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            full_encoded = tokenizer(text, add_special_tokens=True, max_length=2048)
            prompt_encoded = tokenizer(prompt_text, add_special_tokens=True, max_length=2048)

            input_ids = full_encoded["input_ids"]
            attention_mask = full_encoded["attention_mask"]

            # Create labels: -100 for prompt tokens
            labels = input_ids.copy()
            prompt_len = len(prompt_encoded["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        processed_data.append({
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
            "q_values": q_values,
            "generation_order": generation_order,
            "num_candidates": len(candidates)
        })

    return Dataset.from_list(processed_data)


def train_plackett_luce_dpo(
        model_name: str,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
        output_dir: str = "./pl_dpo_output",
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        beta: float = 0.1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        eval_steps: int = 500,
        save_steps: int = 500,
        logging_steps: int = 10,
        warmup_steps: int = 100,
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_wandb: bool = True,
        wandb_project: str = "plackett-luce-dpo",
        gradient_checkpointing: bool = True,
):
    """
    Train Plackett-Luce DPO with LoRA.

    Args:
        model_name: HuggingFace model name
        train_data: List of training samples
        eval_data: Optional list of evaluation samples
        output_dir: Output directory
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        beta: DPO beta parameter
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        warmup_steps: Warmup steps
        max_length: Max sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        gradient_checkpointing: Use gradient checkpointing
    """


    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load reference model (frozen)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model.eval()
    ref_model.requires_grad_(False)

    # Apply LoRA to policy model
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset_for_training(train_data, tokenizer)
    eval_dataset = prepare_dataset_for_training(eval_data, tokenizer) if eval_data else None

    # Training arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = PlackettLuceDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=pl_data_collator
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")

    return model, tokenizer


# Example usage
if __name__ == "__main__":
    # Example training data
    train_data = [
        {
            "messages": [
                {"role": "system", "content": "you are an expert historian"},
                {"role": "user", "content": "who is the father of the nation for India?"}
            ],
            "candidates": [
                {"role": "assistant", "content": "Mahatma Gandhi"},
                {"role": "assistant", "content": "Sachin Tendulkar"},
                {"role": "assistant", "content": "Narendra Modi"},
                {"role": "assistant", "content": "Donald Trump"}
            ],
            "q_values": [0.85, 0.3, 0.0, 0.0],
            "generation_order": [0, 1, 2, 3]
        },
        # Add more samples...
    ]

    eval_data = [
        # Similar format for evaluation
    ]

    # Train
    model, tokenizer = train_plackett_luce_dpo(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        train_data=train_data,
        eval_data=eval_data,
        output_dir="./pl_dpo_lora",
        num_epochs=3,
        learning_rate=5e-5,
        beta=0.1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        eval_steps=500,
        save_steps=500,
        logging_steps=10,
        lora_r=16,
        lora_alpha=32,
        use_wandb=True,
        gradient_checkpointing=True,
    )
