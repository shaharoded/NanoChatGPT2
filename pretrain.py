import os
import time
import json
from contextlib import nullcontext

import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

# Local code
from gpt import GPT
from data.data_load import encode, decode

# Training parameters
EVAL_INTERVAL = 50  # Number of iterations until validation
VALIDATION_SAMPLE_SIZE = 100  # Number of batches for validation
MAX_ITERS = 5000  # Total number of iterations
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP = 1.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'float16' if torch.cuda.is_available() else 'float32'
DEVICE_TYPE = 'cuda' if 'cuda' in DEVICE else 'cpu'
PTDTYPE = {'float32': torch.float32, 'float16': torch.float16}[DTYPE]
CTX = nullcontext() if DEVICE_TYPE == 'cpu' else torch.amp.autocast(device_type=DEVICE_TYPE, dtype=PTDTYPE)

# Model Configuration Path
CONFIG_PATH = 'model_config.json'
DATA_DIR = 'data'


def load_configurations():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def get_model_choice(configs):
    # Display available model configurations and prompt the user to choose
    print("Available model configurations:")
    for idx, model_name in enumerate(configs.keys()):
        print(f"{idx + 1}: {model_name}")
    choice = int(input("Select a model configuration by number: ")) - 1
    model_name = list(configs.keys())[choice]
    return model_name, configs[model_name]


def initialize_model(model_config, model_name):
    out_dir = f'out\{model_name}'
    os.makedirs(out_dir, exist_ok=True)

    # Set seeds and enable TF32 if using CUDA
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Initialize the model
    print(f"[RUNTIME STATUS]: Initializing {model_name} model...")
    model = GPT(model_config)
    model.to(DEVICE)

    return model, out_dir


def load_data():
    # Load training and validation data
    train_data = np.memmap(os.path.join(DATA_DIR, 'pretrain_train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_DIR, 'pretrain_val.bin'), dtype=np.uint16, mode='r')
    return train_data, val_data


def get_batch(data, split, model_config):
    data_split = data['train'] if split == 'train' else data['val']
    ix = torch.randint(len(data_split) - model_config["block_size"], (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data_split[i:i + model_config["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data_split[i + 1:i + 1 + model_config["block_size"]].astype(np.int64)) for i in ix])
    if DEVICE_TYPE == 'cuda':
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


def train_model(model, optimizer, data, model_config, out_dir, model_name):
    scaler = torch.amp.GradScaler(enabled=(DTYPE == 'float16'))
    start_time = time.time()
    best_val_loss = float('inf')
    base_model_path = os.path.join(out_dir, f"{model_name}_base_model.pt")
    print(f"[RUNTIME INFO]: Best model will be saved as {base_model_path}")

    for iter_num in range(MAX_ITERS):
        # Fetch a batch
        X, Y = get_batch(data, 'train', model_config)

        # Forward, backward, and update
        with CTX:
            logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Logging and evaluation
        if iter_num % EVAL_INTERVAL == 0:
            elapsed = time.time() - start_time
            model.eval()
            train_loss = sum(model(*get_batch(data, 'train', model_config))[1].item() for _ in range(VALIDATION_SAMPLE_SIZE)) / VALIDATION_SAMPLE_SIZE
            val_loss = sum(model(*get_batch(data, 'val', model_config))[1].item() for _ in range(VALIDATION_SAMPLE_SIZE)) / VALIDATION_SAMPLE_SIZE
            print(f"[RUNTIME STATUS]: Iter {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, time {(elapsed/60):.2f}m")
            model.train()

            # Save model checkpoint, if validation loss decreased
            if val_loss < best_val_loss:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, base_model_path)
                best_val_loss = val_loss

    print("[RUNTIME STATUS]: Training complete.\n")
    return base_model_path


def load_best_model(model, optimizer, base_model_path):
    # Load the best model to validate loading functionality for later fine-tuning
    print(f"[PRETRAIN SUMMARY]: Loading best model from {base_model_path}")
    checkpoint = torch.load(base_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Generating text from best model
    decoded_text = model.generate_text(context=None, max_new_tokens=200, device=DEVICE)
    print("\nGenerated Text:\n")
    print(decoded_text)


def main():
    # Load configurations
    configs = load_configurations()

    # Get model choice from user
    model_name, model_config = get_model_choice(configs)

    # Initialize model
    model, out_dir = initialize_model(model_config, model_name)

    # Load data
    train_data, val_data = load_data()
    data = {'train': train_data, 'val': val_data}

    # Initialize optimizer
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2), DEVICE_TYPE)

    # Train model
    base_model_path = train_model(model, optimizer, data, model_config, out_dir, model_name)

    # Load best model and generate text
    load_best_model(model, optimizer, base_model_path)


if __name__ == "__main__":
    main()
