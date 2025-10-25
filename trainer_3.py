import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
import gc


def compute_plackett_luce_logprob(scores):
    """
    Compute log probability of a ranking under Plackett-Luce model.

    Args:
        scores: [k] tensor - scores for items in ranked order (best to worst)

    Returns:
        log_prob: tensor scalar - log probability of this ranking
    """
    k = len(scores)
    log_prob = scores.new_zeros(())  # Initialize as tensor on correct device

    for i in range(k - 1):
        remaining_scores = scores[i:]

        # Logsumexp for numerical stability
        max_score = torch.max(remaining_scores)
        logsumexp = max_score + torch.log(torch.sum(torch.exp(remaining_scores - max_score)))

        log_prob = log_prob + scores[i] - logsumexp

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


def compute_sequence_logprobs(logits, labels, label_pad_token_id=-100, length_normalize=True):
    """
    Compute per-sequence log probabilities, properly handling padding.

    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len] with -100 for padding/prompt tokens
        label_pad_token_id: Token ID used for padding (default: -100)
        length_normalize: If True, divide by sequence length (prevents length bias)

    Returns:
        sequence_logps: [batch_size] - sum (or average if normalized) of log probs per sequence
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Create mask BEFORE gathering
    loss_mask = (shift_labels != label_pad_token_id)

    # Replace -100 with 0 for safe gathering (will be masked out anyway)
    gather_labels = shift_labels.clone()
    gather_labels[~loss_mask] = 0

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs of actual tokens
    per_token_logps = torch.gather(
        log_probs,
        dim=2,
        index=gather_labels.unsqueeze(2)
    ).squeeze(2)

    # Apply mask
    per_token_logps = per_token_logps * loss_mask.float()

    # Sum across sequence length
    sequence_logps = per_token_logps.sum(dim=1)

    # Length normalization - divide by number of response tokens
    if length_normalize:
        num_tokens = loss_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
        sequence_logps = sequence_logps / num_tokens

    return sequence_logps


@dataclass
class PlackettLuceCollator:
    """
    Data collator for Plackett-Luce DPO training.
    Handles variable-length sequences and multiple candidates per sample.
    """
    tokenizer: Any
    max_length: int = 512
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        Each sample has k candidates with variable sequence lengths.
        We'll pad and concatenate all candidates from all samples.
        """
        batch_size = len(features)

        # Extract metadata
        all_q_values = []
        all_generation_orders = []
        all_num_candidates = []

        # Collect all candidates across batch
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for feature in features:
            num_candidates = feature["num_candidates"]
            all_num_candidates.append(num_candidates)
            all_q_values.append(feature["q_values"])
            all_generation_orders.append(feature["generation_order"])

            # Truncate and pad each candidate
            for i in range(num_candidates):
                input_ids = feature["input_ids"][i]
                attention_mask = feature["attention_mask"][i]
                labels = feature["labels"][i]

                # Truncate to max_length
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    labels = labels[:self.max_length]

                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_labels.append(labels)

        # Find max length in this batch
        max_len = max(len(ids) for ids in all_input_ids)

        # Pad all sequences
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for input_ids, attention_mask, labels in zip(all_input_ids, all_attention_masks, all_labels):
            pad_len = max_len - len(input_ids)

            padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_masks.append(attention_mask + [0] * pad_len)
            padded_labels.append(labels + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "q_values": all_q_values,
            "generation_order": all_generation_orders,
            "num_candidates": all_num_candidates,
            "batch_size": batch_size,
        }


class MemoryCleanupCallback(TrainerCallback):
    """
    Aggressive memory cleanup callback.
    Clears CUDA cache every N steps.
    """

    def __init__(self, cleanup_every_n_steps: int = 1):
        self.cleanup_every_n_steps = cleanup_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Clean after EVERY step for now
        if state.global_step % self.cleanup_every_n_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()

            # Print memory stats every 10 steps
            if torch.cuda.is_available() and state.global_step % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3
                max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(
                    f"\n[Step {state.global_step}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")
                # Reset peak stats
                torch.cuda.reset_peak_memory_stats()

    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

    def on_log(self, args, state, control, **kwargs):
        # Also clean on log to prevent accumulation
        torch.cuda.empty_cache()


class SaveWithTokenizerCallback(TrainerCallback):
    """
    Save model with tokenizer at specified intervals.
    """

    def __init__(self, tokenizer, save_steps: int = 1000):
        self.tokenizer = tokenizer
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            output_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
            print(f"\nSaving model and tokenizer to {output_dir}")

            # Save model (trainer will handle this)
            control.should_save = True

            # Save tokenizer explicitly
            self.tokenizer.save_pretrained(output_dir)

    def on_save(self, args, state, control, **kwargs):
        # Also save tokenizer whenever model is saved
        output_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        self.tokenizer.save_pretrained(output_dir)


class CumulativeMetricsCallback(TrainerCallback):
    """
    Track cumulative metrics and add them to trainer_state.json.
    """

    def __init__(self):
        self.cumulative_metrics = {
            'total_loss': 0.0,
            'total_rewards_accuracy': 0.0,
            'total_rewards_margin': 0.0,
            'total_logps_margin': 0.0,
            'num_batches': 0,
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update cumulative metrics from logs."""
        if logs is not None:
            # Update cumulative metrics
            if 'loss' in logs:
                self.cumulative_metrics['total_loss'] += logs['loss']
                self.cumulative_metrics['num_batches'] += 1

            if 'rewards/accuracy' in logs:
                self.cumulative_metrics['total_rewards_accuracy'] += logs['rewards/accuracy']

            if 'rewards/margin' in logs:
                self.cumulative_metrics['total_rewards_margin'] += logs['rewards/margin']

            if 'logps/margin' in logs:
                self.cumulative_metrics['total_logps_margin'] += logs['logps/margin']

            # Compute averages
            n = self.cumulative_metrics['num_batches']
            if n > 0:
                cumulative_avg = {
                    'cumulative/avg_loss': self.cumulative_metrics['total_loss'] / n,
                    'cumulative/avg_rewards_accuracy': self.cumulative_metrics['total_rewards_accuracy'] / n,
                    'cumulative/avg_rewards_margin': self.cumulative_metrics['total_rewards_margin'] / n,
                    'cumulative/avg_logps_margin': self.cumulative_metrics['total_logps_margin'] / n,
                    'cumulative/num_batches': n,
                }

                # Add to state's log_history (will be saved in trainer_state.json)
                if state.log_history:
                    state.log_history[-1].update(cumulative_avg)


class PlackettLuceDPOTrainer(Trainer):
    """
    Trainer for Plackett-Luce DPO with k > 2 candidates.
    Extends base Trainer for full control.
    """

    def __init__(
            self,
            model,
            ref_model,
            beta: float = 0.1,
            length_normalize: bool = True,
            *args,
            **kwargs
    ):
        super().__init__(model=model, *args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.length_normalize = length_normalize

        # Set reference model to eval mode and freeze
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        # Move ref model to same device as policy model
        if hasattr(self.model, 'device'):
            self.ref_model = self.ref_model.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute Plackett-Luce DPO loss.
        Handles variable k across batch samples.
        """
        # Extract batch info
        batch_size = inputs["batch_size"]
        num_candidates_list = inputs["num_candidates"]

        # Forward pass through policy model
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        policy_logits = outputs.logits

        # Forward pass through reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            ref_logits = ref_outputs.logits

        # Compute sequence log probabilities for ALL candidates (flattened)
        policy_sequence_logps = compute_sequence_logprobs(
            policy_logits,
            inputs["labels"],
            self.args.label_pad_token_id if hasattr(self.args, 'label_pad_token_id') else -100
        )

        ref_sequence_logps = compute_sequence_logprobs(
            ref_logits,
            inputs["labels"],
            self.args.label_pad_token_id if hasattr(self.args, 'label_pad_token_id') else -100
        )

        # Split logprobs by sample (handle variable k)
        losses = []
        all_metrics = {
            'logprobs_policy_ranking': [],
            'logprobs_ref_ranking': [],
            'logprobs_margin': [],
            'rewards_margin': [],
            'rewards_accuracy': [],
        }

        offset = 0
        for i in range(batch_size):
            k = num_candidates_list[i]

            # Extract logprobs for this sample's k candidates
            policy_logps = policy_sequence_logps[offset:offset + k]
            ref_logps = ref_sequence_logps[offset:offset + k]
            offset += k

            # Get ranking for this sample
            q_values = inputs["q_values"][i]
            generation_order = inputs["generation_order"][i]

            ranking_indices = get_ranking_from_q_and_genorder(q_values, generation_order)

            # CRITICAL: Don't create new tensor, use indexing directly
            # This preserves gradient flow better
            ranking_indices_list = ranking_indices if isinstance(ranking_indices, list) else ranking_indices.tolist()

            # Reorder logprobs according to ranking - use list indexing to preserve grads
            policy_scores_ranked = torch.stack([policy_logps[idx] for idx in ranking_indices_list])
            ref_scores_ranked = torch.stack([ref_logps[idx] for idx in ranking_indices_list])

            # Compute Plackett-Luce log probabilities
            log_p_policy = compute_plackett_luce_logprob(policy_scores_ranked)
            log_p_ref = compute_plackett_luce_logprob(ref_scores_ranked)

            # DPO loss
            logits = self.beta * (log_p_policy - log_p_ref)
            loss = -F.logsigmoid(logits)

            losses.append(loss)

            # Collect metrics
            all_metrics['logprobs_policy_ranking'].append(log_p_policy.item())
            all_metrics['logprobs_ref_ranking'].append(log_p_ref.item())
            all_metrics['logprobs_margin'].append((log_p_policy - log_p_ref).item())
            all_metrics['rewards_margin'].append(logits.item())
            all_metrics['rewards_accuracy'].append((logits > 0).float().item())

        # Average loss
        if len(losses) == 0:
            print("[ERROR] No valid losses computed in this batch!")
            # Return a dummy loss to avoid crash
            return torch.tensor(0.0, requires_grad=True, device=model.device)

        loss = torch.stack(losses).mean()

        # Log averaged metrics
        for key, values in all_metrics.items():
            self.log({key: sum(values) / len(values)})

        return (loss, outputs) if return_outputs else loss


def prepare_dataset_for_training(data_list: List[Dict], tokenizer, max_length: int = 512) -> Dataset:
    """
    Prepare dataset in the format expected by PlackettLuceDPOTrainer.

    Args:
        data_list: List of dicts with "messages", "candidates", "q_values", "generation_order"
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

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

            # Tokenize WITHOUT truncation first
            full_encoded = tokenizer(text, add_special_tokens=True, truncation=False)
            prompt_encoded = tokenizer(prompt_text, add_special_tokens=True, truncation=False)

            input_ids = full_encoded["input_ids"]
            attention_mask = full_encoded["attention_mask"]
            prompt_len = len(prompt_encoded["input_ids"])

            # Create labels: -100 for prompt tokens
            labels = input_ids.copy()
            labels[:prompt_len] = [-100] * prompt_len

            # NOW truncate if needed - from LEFT to preserve response
            if len(input_ids) > max_length:
                # Calculate how much to remove
                overflow = len(input_ids) - max_length

                # Truncate from left (remove from prompt, keep response)
                input_ids = input_ids[overflow:]
                attention_mask = attention_mask[overflow:]
                labels = labels[overflow:]

                # Note: After truncation, some prompt tokens might remain at start
                # Those will still be marked as -100, which is correct

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
        max_steps: Optional[int] = None,
        learning_rate: float = 5e-5,
        beta: float = 0.1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        eval_steps: int = 500,
        save_strategy: str = "steps",
        save_steps: int = 1000,
        logging_steps: int = 10,
        warmup_steps: int = 100,
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
        memory_cleanup_steps: int = 5,
        load_in_8bit: bool = False,
):
    """
    Train Plackett-Luce DPO with LoRA.

    Memory optimization tips:
    - Reduce max_length (e.g., 256 or 384 instead of 512)
    - Enable load_in_8bit=True for quantization
    - Reduce per_device_train_batch_size to 1
    - Increase gradient_accumulation_steps
    - Reduce lora_r (e.g., 8 instead of 16)
    """

    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models with optional 8-bit quantization
    print(f"Loading model from {model_name}...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        print("Loading model in 8-bit mode for memory efficiency...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

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

    if gradient_checkpointing and not load_in_8bit:
        model.gradient_checkpointing_enable()

    if load_in_8bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_config)
    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset_for_training(train_data, tokenizer, max_length)
    eval_dataset = prepare_dataset_for_training(eval_data, tokenizer, max_length) if eval_data else None

    # Data collator
    data_collator = PlackettLuceCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        label_pad_token_id=-100
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps else -1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=gradient_checkpointing and not load_in_8bit,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,  # Keep only 2 checkpoints to save disk
        label_names=["labels"],
        logging_first_step=True,
        dataloader_pin_memory=False,  # Disable pin memory to save RAM
        max_grad_norm=1.0,  # Gradient clipping
    )

    # Initialize callbacks
    memory_callback = MemoryCleanupCallback(cleanup_every_n_steps=1)  # Clean EVERY step
    save_tokenizer_callback = SaveWithTokenizerCallback(tokenizer=tokenizer, save_steps=save_steps)
    cumulative_metrics_callback = CumulativeMetricsCallback()

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = PlackettLuceDPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=beta,
        length_normalize=True,  # CRITICAL: Enable length normalization
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[memory_callback, save_tokenizer_callback, cumulative_metrics_callback],
    )

    # Train
    print("\nStarting training...")
    print(f"Total training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Total eval samples: {len(eval_dataset)}")
    print(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    if max_steps:
        print(f"Max steps: {max_steps}")
    else:
        print(f"Epochs: {num_epochs}")
    print(f"Beta: {beta}")
    print(f"Max sequence length: {max_length}")
    print(f"LoRA rank: {lora_r}")
    print(f"8-bit quantization: {load_in_8bit}")
    print(f"Save model + tokenizer every {save_steps} steps")
    print("=" * 60)

    trainer.train()

    # Save final model and tokenizer
    print(f"\nSaving final model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Clean up
    torch.cuda.empty_cache()
    gc.collect()

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

    eval_data = []  # Similar format

    # Train
    model, tokenizer = train_plackett_luce_dpo(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        train_data=train_data,
        eval_data=eval_data,
        output_dir="./pl_dpo_lora",
        num_epochs=3,
        max_steps=None,
        learning_rate=5e-5,
        beta=0.1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        eval_steps=500,
        save_strategy="steps",  # Save by steps
        save_steps=1000,  # Save every 1000 steps
        logging_steps=10,
        lora_r=16,
        lora_alpha=32,
        gradient_checkpointing=True,
        memory_cleanup_steps=10,
    )
