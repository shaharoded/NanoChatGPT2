import numpy as np
import random
import torch
from torch.nn import functional as F
import os
import time
import json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

# Local code
from data.data_load import encode

from train_utils import (
    DEVICE, DEVICE_TYPE, DATA_DIR,
    initialize_model, load_configurations, get_model_choice
)

# Directory where base models are saved
OUT_DIR = "out"
CONFIG_PATH = 'model_config.json'
QA_TRAIN_PATH = os.path.join(DATA_DIR, "qa_train.json")
QA_VAL_PATH = os.path.join(DATA_DIR, "qa_val.json")

# Fine-tuning parameters (optimized for QA data)
FT_LEARNING_RATE = 1e-5  # Smaller learning rate for fine-tuning
FT_WEIGHT_DECAY = 0.005  # Different weight decay for fine-tuning
FT_BETA1, FT_BETA2 = 0.9, 0.98
GRAD_CLIP = 1.0

EVAL_ITERS = 200
EPOCHS = 8

'''
The model is limited to sequence length == model.block_size, so batch size should be small enough to ensure
no longer sequences are passed.
QA data has ~80K pairs. I'll feed them question by question while ensuring each pair is not too long for the batch. 
On this module I'll work by EPOCHS and not ITERATIONS (concatenated block_size text)
'''


def load_qa_data():
    """Load QA train and validation data from JSON files."""
    with open(QA_TRAIN_PATH, 'r') as f:
        qa_train_data = json.load(f)  # List of tokenized QA pairs
    with open(QA_VAL_PATH, 'r') as f:
        qa_val_data = json.load(f)  # List of tokenized QA pairs
    return qa_train_data, qa_val_data


def finetune_model(model, optimizer, data, model_name, block_size):
    """
    Fine-tune the model on the QA dataset using epochs.
    Args:
        model: The GPT model to be fine-tuned.
        optimizer: The optimizer for the model.
        data (dict): Dictionary containing 'train' and 'val' splits, each is a list of dictionaries, each dictionary is a pair.
        model_name (str): Name of the model for saving checkpoints.
        block_size (int): Maximum sequence length for input + output.
    """
    print("[RUNTIME STATUS]: Starting fine-tuning on QA data...")
    start_time = time.time()
    best_val_loss = float('inf')
    checkpoint_path = None

    train_data = data['train']
    val_data = data['val']

    for epoch in range(EPOCHS):
        random.shuffle(train_data)  # Shuffle data at the start of each epoch
        valid_pairs = 0  # Track valid QA pairs processed in this epoch

        for pair in train_data:
            # Prepare input and output tensors
            input_tensor = torch.tensor(pair['input'], dtype=torch.int64).to(DEVICE)
            output_tensor = torch.tensor(pair['output'], dtype=torch.int64).to(DEVICE)

            # Combine input and output for full sequence
            full_sequence = torch.cat([input_tensor, output_tensor], dim=0).unsqueeze(0)  # Add batch dimension

            # Skip pairs exceeding the block size
            if full_sequence.size(1) > block_size:
                continue

            valid_pairs += 1

            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(full_sequence)

            # Extract logits corresponding to the output tokens
            logits_for_loss = logits[:, -output_tensor.size(0):, :]  # Only the output token logits
            loss = F.cross_entropy(
                logits_for_loss.view(-1, logits.size(-1)),  # Flatten logits
                output_tensor.view(-1)  # Flatten targets
            )

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # Log every EVAL_ITERS
            if valid_pairs % EVAL_ITERS == 0:
                elapsed = time.time() - start_time
                model.eval()
                val_loss = sum(
                    F.cross_entropy(
                        model(
                            torch.tensor(pair['input'] + pair['output'], dtype=torch.int64).unsqueeze(0).to(DEVICE),
                            torch.tensor(pair['output'], dtype=torch.int64).unsqueeze(0).to(DEVICE)
                        )[0][:, -len(pair['output']):, :].view(-1, logits.size(-1)),  # Flatten logits
                        torch.tensor(pair['output'], dtype=torch.int64).to(DEVICE).view(-1)  # Flatten targets
                    ).item()
                    for pair in val_data
                ) / len(val_data)

                print(
                    f"[RUNTIME STATUS]: EPOCH: {epoch + 1}/{EPOCHS}, Pairs: {valid_pairs}, Val loss: {val_loss:.4f}, Time: {elapsed / 60:.2f}m"
                )

                # Save the model if the validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(OUT_DIR, model_name, f"{model_name}_fine_tuned.pt")
                    torch.save(model.state_dict(), checkpoint_path)

                model.train()

        if valid_pairs == 0:
            raise ValueError(
                f"[ERROR]: No valid QA pairs processed in epoch {epoch + 1}. Consider increasing block size or dataset size."
            )

    print("[RUNTIME STATUS]: Fine-tuning complete.")
    if checkpoint_path:
        print(f"[RUNTIME STATUS]: Best model saved to {checkpoint_path}")

    return checkpoint_path


def generate_examples(model):
    """Generate examples to see if fine-tuning worked."""
    questions = [
        "What is the capital of France?",
        "How many planets are in our solar system?",
        "Who wrote the novel '1984'?",
        "What is the speed of light?",
        "Who is the president of the United States?"
    ]

    print("\n[TEXT GENERATION EXAMPLES]:")
    model.eval()  # Ensure model is in evaluation mode
    for question in questions:
        input_tokens = encode(question)
        input_tensor = torch.tensor(input_tokens, dtype=torch.int64).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            answer = model.generate_text(input_tensor, max_new_tokens=50, device=DEVICE)
        print(f"Q: {question}\nA: {answer}\n")


def main():
    # Load configurations and initialize model
    model_directories = [d for d in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, d))]
    configs = load_configurations()

    # Filter configurations that match the model directories
    configs = {model_name: config for model_name, config in configs.items() if model_name in model_directories}

    # Get model choice from the available configurations
    model_name, model_config = get_model_choice(configs)
    block_size = model_config.get('block_size', 0)
    # Initialize model
    model, optimizer, _ = initialize_model(model_config=model_config,
                                                model_name=model_name,
                                                step='fine_tuned',
                                                learning_rate=FT_LEARNING_RATE,
                                                weight_decay=FT_WEIGHT_DECAY,
                                                betas=(FT_BETA1, FT_BETA2))

    # Initialize optimizer with fine-tuning parameters
    optimizer = model.configure_optimizers(FT_WEIGHT_DECAY, FT_LEARNING_RATE, (FT_BETA1, FT_BETA2), DEVICE_TYPE)

    # Load QA data
    qa_train_data, qa_val_data = load_qa_data()
    data = {'train': qa_train_data, 'val': qa_val_data}

    # Fine-tune the model
    checkpoint_path = finetune_model(model, optimizer, data, model_name, block_size)

    # Load the best model and generate examples
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
    generate_examples(model)


if __name__ == "__main__":
    main()
