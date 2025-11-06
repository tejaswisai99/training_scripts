import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Dict, List, Any
import json
from pathlib import Path


# ============================================================================
# Core Functions (from your original code)
# ============================================================================

def placket_luce_loss(scores):
    """
    Compute Plackett-Luce log probability for a ranking.
    
    Args:
        scores: [k] tensor of scores for k candidates (ordered by rank)
    
    Returns:
        log_prob: scalar tensor - log probability under Plackett-Luce model
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


def get_sequence_logprob(model, input_ids, attention_mask, response_start_idx):
    """
    Compute log probability of the response part of a sequence.

    Args:
        model: The language model
        input_ids: [1, seq_len] - full sequence (messages + candidate)
        attention_mask: [1, seq_len] - attention mask
        response_start_idx: int - index where assistant response starts

    Returns:
        logprob: scalar - sum of log probabilities for the response tokens
    """
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [1, seq_len, vocab_size]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()  # [1, seq_len-1, vocab_size]
    shift_labels = input_ids[:, 1:].contiguous()  # [1, seq_len-1]

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [1, seq_len-1, vocab_size]

    # Gather log probs of actual tokens
    per_token_logps = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.unsqueeze(2)
    ).squeeze(2)  # [1, seq_len-1]

    # Only sum log probs for response tokens (from response_start_idx onward)
    response_mask = torch.zeros_like(per_token_logps)
    if response_start_idx > 0:
        response_mask[:, response_start_idx - 1:] = 1.0

    masked_logps = per_token_logps * response_mask
    # Sum of log-probs and count of response tokens
    sequence_logprob = masked_logps.sum()
    num_response_tokens = response_mask.sum().clamp(min=1)  # avoid div/0

    # Length normalization
    normalized_logprob = sequence_logprob / num_response_tokens

    return normalized_logprob


def compute_candidate_logprobs(model, tokenizer, messages, candidates):
    """
    Compute log probabilities for all candidates given messages.

    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: list of message dicts with "role" and "content"
        candidates: list of candidate dicts with "role" and "content"

    Returns:
        logprobs: [k] tensor - log probability for each candidate
    """
    logprobs = []

    for candidate in candidates:
        # Create full conversation: messages + this candidate
        full_conversation = messages + [candidate]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            full_conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        # Also get the text without the candidate to find where response starts
        text_without_candidate = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize full sequence
        encoded = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Tokenize without candidate to find response start position
        encoded_without = tokenizer(text_without_candidate, return_tensors="pt")
        response_start_idx = encoded_without["input_ids"].shape[1]

        # Compute log probability
        logprob = get_sequence_logprob(
            model,
            input_ids,
            attention_mask,
            response_start_idx
        )
        logprobs.append(logprob)

    # Stack into tensor [k]
    logprobs_tensor = torch.stack(logprobs)

    return logprobs_tensor


# ============================================================================
# Dataset Class
# ============================================================================

class PlackettLuceDataset(torch.utils.data.Dataset):
    """
    Dataset for Plackett-Luce ranking training.
    
    Expected data format:
    {
        "messages": [...],
        "candidates": [...],
        "q_values": [1.0, 0.0, 0.0],  # Ground truth ranking scores
        "generation_order": [0, 1, 2]
    }
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to JSONL file where each line is a training example
        """
        self.examples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================================
# Custom Data Collator
# ============================================================================

@dataclass
class PlackettLuceDataCollator:
    """
    Custom data collator that doesn't tokenize - just passes data through.
    The actual tokenization happens in the compute_loss function.
    """
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simply batch the examples without tokenization.
        
        Args:
            features: List of examples from the dataset
            
        Returns:
            Dict with batched data
        """
        return {
            "examples": features
        }


# ============================================================================
# Custom Trainer
# ============================================================================

class PlackettLuceTrainer(Trainer):
    """
    Custom Trainer that computes Plackett-Luce loss.
    """
    
    def __init__(self, ref_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.eval()
            # Freeze reference model
            for param in self.ref_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation using Plackett-Luce model.
        
        Args:
            model: The model being trained
            inputs: Dict with "examples" key containing list of examples
            return_outputs: Whether to return model outputs
            
        Returns:
            loss: The computed loss
        """
        examples = inputs["examples"]
        total_loss = 0.0
        num_examples = len(examples)
        
        for example in examples:
            messages = example["messages"]
            candidates = example["candidates"]
            q_values = torch.tensor(example["q_values"], device=model.device)
            
            # Sort candidates by q_values (descending) to get true ranking
            sorted_indices = torch.argsort(q_values, descending=True)
            sorted_candidates = [candidates[i] for i in sorted_indices]
            
            # Compute log probabilities from policy model
            policy_logprobs = compute_candidate_logprobs(
                model, 
                self.tokenizer, 
                messages, 
                sorted_candidates
            )
            
            # Compute Plackett-Luce log probability (higher is better)
            pl_log_prob = placket_luce_loss(policy_logprobs)
            
            # If using reference model, compute KL penalty
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_logprobs = compute_candidate_logprobs(
                        self.ref_model,
                        self.tokenizer,
                        messages,
                        sorted_candidates
                    )
                
                # KL divergence penalty
                kl_penalty = (policy_logprobs - ref_logprobs).sum()
                beta = 0.1  # KL penalty coefficient
                
                # Loss is negative log prob plus KL penalty
                loss = -pl_log_prob + beta * kl_penalty
            else:
                # Loss is just negative log probability
                loss = -pl_log_prob
            
            total_loss += loss
        
        # Average loss over batch
        avg_loss = total_loss / num_examples
        
        return (avg_loss, None) if return_outputs else avg_loss


# ============================================================================
# Training Setup Functions
# ============================================================================

def setup_lora_model(model, lora_config_dict=None):
    """
    Apply LoRA to the model.
    
    Args:
        model: Base model
        lora_config_dict: Optional dict with LoRA configuration
        
    Returns:
        model: Model with LoRA applied
    """
    if lora_config_dict is None:
        lora_config_dict = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }
    
    lora_config = LoraConfig(**lora_config_dict)
    model = get_peft_model(model, lora_config)
    
    print(f"\n{'='*60}")
    print("LoRA Configuration Applied")
    print(f"{'='*60}")
    model.print_trainable_parameters()
    print(f"{'='*60}\n")
    
    return model


def load_reference_model(model_name):
    """
    Load a frozen reference model for KL penalty computation.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        ref_model: Frozen reference model
    """
    print(f"Loading reference model: {model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    ref_model.eval()
    
    # Freeze all parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    return ref_model


# ============================================================================
# Main Training Function
# ============================================================================

def train_plackett_luce(
    model_name: str,
    train_data_path: str,
    eval_data_path: str = None,
    output_dir: str = "./plackett_luce_output",
    use_lora: bool = True,
    lora_config: dict = None,
    use_reference_model: bool = True,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 3,
):
    """
    Main training function.
    
    Args:
        model_name: HuggingFace model name or path
        train_data_path: Path to training data JSONL file
        eval_data_path: Optional path to evaluation data JSONL file
        output_dir: Directory to save model checkpoints
        use_lora: Whether to use LoRA
        lora_config: Optional LoRA configuration dict
        use_reference_model: Whether to use a reference model for KL penalty
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        save_total_limit: Maximum number of checkpoints to keep
    """
    
    print(f"\n{'='*60}")
    print("Plackett-Luce Training Setup")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Apply LoRA if requested
    if use_lora:
        model = setup_lora_model(model, lora_config)
    
    # Load reference model if requested
    ref_model = None
    if use_reference_model:
        ref_model = load_reference_model(model_name)
    
    # Load datasets
    print(f"\nLoading training data from: {train_data_path}")
    train_dataset = PlackettLuceDataset(train_data_path)
    print(f"Training examples: {len(train_dataset)}")
    
    eval_dataset = None
    if eval_data_path:
        print(f"Loading evaluation data from: {eval_data_path}")
        eval_dataset = PlackettLuceDataset(eval_data_path)
        print(f"Evaluation examples: {len(eval_dataset)}")
    
    # Create data collator
    data_collator = PlackettLuceDataCollator(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if eval_dataset else None,
        save_total_limit=save_total_limit,
        evaluation_strategy="steps" if eval_dataset else "no",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        report_to=["tensorboard"],
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,  # Important: we handle data manually
    )
    
    # Create trainer
    trainer = PlackettLuceTrainer(
        ref_model=ref_model,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    trainer.train()
    
    # Save final model
    print(f"\n{'='*60}")
    print("Saving Final Model")
    print(f"{'='*60}\n")
    
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    print(f"\nTraining complete! Model saved to: {output_dir}/final_model")
    
    return trainer, model


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create a sample training data file
    sample_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "candidates": [
            {"role": "assistant", "content": "The answer is 4."},
            {"role": "assistant", "content": "I don't know."},
            {"role": "assistant", "content": "The answer is 5."}
        ],
        "q_values": [1.0, 0.5, 0.0],
        "generation_order": [0, 1, 2]
    }
    
    # Create sample data file
    Path("sample_train_data.jsonl").write_text(json.dumps(sample_data) + "\n")
    
    # Run training
    trainer, model = train_plackett_luce(
        model_name="Qwen/Qwen3-0.6B",
        train_data_path="sample_train_data.jsonl",
        output_dir="./pl_training_output",
        use_lora=True,
        use_reference_model=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=10,
        logging_steps=5,
        save_steps=100,
    )
