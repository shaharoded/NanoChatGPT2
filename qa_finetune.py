import numpy as np
import random
import torch
from torch.nn import functional as F
import os
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

# Local code
from data.data_load import encode, TOKENIZER

from train_utils import (
    DEVICE, DTYPE, CTX, DATA_DIR,
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

MAX_ITERS = 1000
BATCH_SIZE = 8
VALIDATION_SAMPLE_SIZE = 100
EVAL_INTERVAL = 25


def load_qa_data():
    # Load training and validation data
    train_data = np.memmap(os.path.join(DATA_DIR, 'qa_train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_DIR, 'qa_val.bin'), dtype=np.uint16, mode='r')
    return train_data, val_data

def filter_and_trim_qa_stream(data_stream, block_size):
    """
    Filter out long QA pairs and trim the tokenized stream to only valid QA pairs.
    
    Args:
        data_stream (np.array): The continuous tokenized stream.
        block_size (int): Maximum sequence length for a model input.
    
    Returns:
        tuple: A filtered and trimmed tokenized stream and valid start indices.
    
    Raises:
        ValueError: If the number of valid QA pairs is less than VALIDATION_SAMPLE_SIZE.
    """
    filtered_stream = []
    valid_indices = []
    current_start = 0
    valid_pairs_count = 0

    for i, token in enumerate(data_stream):
        if token == TOKENIZER.eot_token:
            length = i - current_start + 1  # Include the <EOT> token
            if length <= block_size:
                # Include valid QA pair in the filtered stream
                filtered_stream.extend(data_stream[current_start:i + 1])
                valid_indices.append(current_start)
                valid_pairs_count += 1
            # Move to the next QA pair
            current_start = i + 1

    # Check if we have enough QA pairs for evaluation
    if valid_pairs_count < VALIDATION_SAMPLE_SIZE:
        raise ValueError(
            f"[ERROR]: Not enough QA pairs to evaluate the model. Found {valid_pairs_count} pairs, "
            f"but require at least {VALIDATION_SAMPLE_SIZE}. Consider increasing block size or use shorter QAs."
        )

    print(f"[DATA FILTER]: Retained {valid_pairs_count} valid QA pairs within block_size")
    return np.array(filtered_stream, dtype=np.int64), valid_indices


def pad_to_block_size(tensor, block_size, pad_token=TOKENIZER.eot_token):
    padding = block_size - tensor.size(0)
    if padding > 0:
        tensor = F.pad(tensor, (0, padding), value=pad_token)
    return tensor


def get_batch(data_stream, valid_indices, block_size):
    """
    Get a batch of data for fine-tuning, ensuring QA boundaries are respected.
    Args:
        data_stream (np.array): The continuous tokenized stream.
        valid_indices (list): Precomputed list of valid start indices.
        block_size (int): Maximum sequence length for the model.
    
    Returns:
        tuple: (input_tensor, target_tensor) for the model.
    """
    batch_start_indices = random.choices(valid_indices, k=BATCH_SIZE)

    # Prepare the batch
    input_batch = []
    target_batch = []

    for start_idx in batch_start_indices:
        end_idx = min(start_idx + block_size, len(data_stream))
        sequence = data_stream[start_idx:end_idx]
        sequence = pad_to_block_size(torch.tensor(sequence, dtype=torch.int64), block_size)
        input_batch.append(sequence[:-1])  # Input (all but the last token)
        target_batch.append(sequence[1:])  # Target (all but the first token)

    input_tensor = torch.stack(input_batch).to(DEVICE)
    target_tensor = torch.stack(target_batch).to(DEVICE)
    return input_tensor, target_tensor


def finetune_model(model, optimizer, data, model_config, out_dir, model_name):
    """
    Fine-tune the model on QA data.
    Args:
        model: The GPT model to be fine-tuned.
        optimizer: Optimizer for the model.
        data (dict): Dictionary containing 'train' and 'val' data streams and valid indices.
        model_config (dict): Configuration for the model.
        out_dir (str): Directory for saving model checkpoints.
        model_name (str): Name of the model for checkpointing.
    
    Returns:
        str: Path to the best model checkpoint.
    """
    scaler = torch.amp.GradScaler(enabled=(DTYPE == 'float16'))
    start_time = time.time()
    best_val_loss = float('inf')
    ft_model_path = os.path.join(out_dir, f"{model_name}_fine_tuned.pt")
    print(f"[RUNTIME INFO]: Best model will be saved as {ft_model_path}")

    block_size = model_config['block_size']

    for iter_num in range(MAX_ITERS):
        # Fetch a training batch
        X, Y = get_batch(data['train']['stream'], data['train']['indices'], block_size)

        # Forward, backward, and update
        with CTX:
            logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Logging and validation
        if iter_num % EVAL_INTERVAL == 0:
            elapsed = time.time() - start_time
            model.eval()

            # Compute train and validation losses
            train_loss = sum(
                model(*get_batch(data['train']['stream'], data['train']['indices'], block_size))[1].item()
                for _ in range(VALIDATION_SAMPLE_SIZE)
            ) / VALIDATION_SAMPLE_SIZE
            val_loss = sum(
                model(*get_batch(data['val']['stream'], data['val']['indices'], block_size))[1].item()
                for _ in range(VALIDATION_SAMPLE_SIZE)
            ) / VALIDATION_SAMPLE_SIZE

            print(f"[RUNTIME STATUS]: Iter {iter_num}: train loss {train_loss:.3f}, val loss {val_loss:.3f}, time {(elapsed / 60):.1f}m")
            
            # Save model checkpoint if validation loss improves
            if val_loss < best_val_loss:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, ft_model_path)
                best_val_loss = val_loss

            model.train()

    print("[RUNTIME STATUS]: Fine-tuning complete.")
    return ft_model_path


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
        print(f"Q: {question}\nA: {answer[len(question):]}\n")


def main():
    # Load configurations and initialize model
    model_directories = [d for d in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, d))]
    configs = load_configurations()

    # Filter configurations that match the model directories
    configs = {model_name: config for model_name, config in configs.items() if model_name in model_directories}

    # Get model choice from the available configurations
    model_name, model_config = get_model_choice(configs)
    block_size = model_config.get('block_size', 0)
    if block_size == 0:
        raise ValueError("[ERROR]: Block size must be specified in the model configuration.")

    # Initialize model
    model, optimizer, out_dir = initialize_model(
        model_config=model_config,
        model_name=model_name,
        step='fine_tuned',
        learning_rate=FT_LEARNING_RATE,
        weight_decay=FT_WEIGHT_DECAY,
        betas=(FT_BETA1, FT_BETA2)
    )

    # Load QA data as binary streams
    qa_train_data, qa_val_data = load_qa_data()

    # Filter and trim data streams
    print("[RUNTIME STATUS]: Filtering and trimming QA training data...")
    qa_train_stream, train_indices = filter_and_trim_qa_stream(qa_train_data, block_size)
    print("[RUNTIME STATUS]: Filtering and trimming QA validation data...")
    qa_val_stream, val_indices = filter_and_trim_qa_stream(qa_val_data, block_size)

    # Ensure data streams are not empty
    if len(train_indices) == 0:
        raise ValueError("[ERROR]: No valid training QA pairs found. Please check your data or block size.")
    if len(val_indices) == 0:
        raise ValueError("[ERROR]: No valid validation QA pairs found. Please check your data or block size.")

    # Create the data dictionary
    data = {
        'train': {'stream': qa_train_stream, 'indices': train_indices},
        'val': {'stream': qa_val_stream, 'indices': val_indices}
    }

    # Fine-tune the model
    print("[RUNTIME STATUS]: Starting fine-tuning process...")
    checkpoint_path = finetune_model(
        model=model,
        optimizer=optimizer,
        data=data,
        model_config=model_config,
        out_dir=out_dir,
        model_name=model_name
    )

    # Load the best model and generate examples
    if checkpoint_path:
        print(f"[RUNTIME STATUS]: Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

        print("[RUNTIME INFO]: Generating example outputs...")
        generate_examples(model)
    else:
        print("[RUNTIME WARNING]: Fine-tuning did not complete successfully. Skipping example generation.")


if __name__ == "__main__":
    main()
