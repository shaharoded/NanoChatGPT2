import numpy as np
import torch
import os
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

# Local code
from data.data_load import encode
from pretrain import (
    BETA1, BETA2, DEVICE, DEVICE_TYPE, GRAD_CLIP,
    load_configurations, get_model_choice, initialize_model
)

# Directory where base models are saved
OUT_DIR = "out"
CONFIG_PATH = 'model_config.json'
QA_TRAIN_PATH = os.path.join("data", "qa_train.bin")
QA_VAL_PATH = os.path.join("data", "qa_val.bin")

# Fine-tuning parameters (optimized for QA data)
FT_LEARNING_RATE = 1e-5  # Smaller learning rate for fine-tuning
FT_WEIGHT_DECAY = 0.005  # Different weight decay for fine-tuning
FT_BETA1, FT_BETA2 = 0.9, 0.98

BATCH_SIZE = 4  # QA data is smaller, thus the model might benefit from smaller batches
EVAL_INTERVAL = 25
EVAL_ITERS = 200
MAX_ITERS = 5000


def load_qa_data():
    """Load QA train and validation data from bin files."""
    qa_train_data = np.memmap(QA_TRAIN_PATH, dtype=np.uint16, mode='r')
    qa_val_data = np.memmap(QA_VAL_PATH, dtype=np.uint16, mode='r')
    return qa_train_data, qa_val_data


def get_qa_batch(data, split, model_config):
    """Fetch a batch for QA fine-tuning."""
    data_split = data['train'] if split == 'train' else data['val']
    ix = torch.randint(len(data_split) - model_config["block_size"], (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data_split[i:i + model_config["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data_split[i + 1:i + 1 + model_config["block_size"]].astype(np.int64)) for i in ix])
    
    if DEVICE_TYPE == 'cuda':
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y


def finetune_model(model, optimizer, data, model_config, model_name):
    """Fine-tune the model on the QA dataset."""
    print("[RUNTIME STATUS]: Starting fine-tuning on QA data...")
    start_time = time.time()
    best_val_loss = float('inf')
    checkpoint_path = None

    for iter_num in range(MAX_ITERS):
        # Fetch a batch
        X, Y = get_qa_batch(data, 'train', model_config)

        # Forward, backward, and update
        optimizer.zero_grad()
        logits, loss = model(X, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Evaluate and log
        if iter_num % EVAL_INTERVAL == 0:
            elapsed = time.time() - start_time
            model.eval()
            train_loss = sum(model(*get_qa_batch(data, 'train', model_config))[1].item() for _ in range(EVAL_ITERS)) / EVAL_ITERS
            val_loss = sum(model(*get_qa_batch(data, 'val', model_config))[1].item() for _ in range(EVAL_ITERS)) / EVAL_ITERS
            print(f"[RUNTIME STATUS]: Iter {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, time {(elapsed/60):.2f}m")

            # Save the model if the validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(OUT_DIR, model_name, f"{model_name}_fine_tuned.pt")
                torch.save(model.state_dict(), checkpoint_path)

            model.train()

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

    # Initialize model
    model, _ = initialize_model(model_config, model_name)

    # Initialize optimizer with fine-tuning parameters
    optimizer = model.configure_optimizers(FT_WEIGHT_DECAY, FT_LEARNING_RATE, (FT_BETA1, FT_BETA2), DEVICE_TYPE)

    # Load QA data
    qa_train_data, qa_val_data = load_qa_data()
    data = {'train': qa_train_data, 'val': qa_val_data}

    # Fine-tune the model
    checkpoint_path = finetune_model(model, optimizer, data, model_config, model_name)

    # Load the best model and generate examples
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
    generate_examples(model)


if __name__ == "__main__":
    main()
