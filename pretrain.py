import os
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

# Local code
from train_utils import (
    DEVICE, DEVICE_TYPE, DTYPE, CTX, DATA_DIR,
    initialize_model, load_configurations, get_model_choice, cpu_memory_usage
)

# Training parameters
EVAL_INTERVAL = 25  # Number of iterations until validation
VALIDATION_SAMPLE_SIZE = 100  # Number of batches for validation
MAX_ITERS = 2000  # Total number of iterations
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
BETA1, BETA2 = 0.9, 0.98
GRAD_CLIP = 1.0

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
    
    # Add the scheduler (for learning rate decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for iter_num in range(MAX_ITERS):
        # Monitor CPU memory usage
        mem_use = cpu_memory_usage()
        
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
            print(f"[RUNTIME STATUS]: Iter {iter_num}: train loss {train_loss:.3f}, val loss {val_loss:.3f}, time {(elapsed/60):.1f}m, Memory Usage: {mem_use}")
            scheduler.step(val_loss)    # Step the scheduler with validation loss
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
    model, optimizer, out_dir = initialize_model(model_config=model_config,
                                                 model_name=model_name,
                                                 step='base_model',
                                                 learning_rate=LEARNING_RATE,
                                                 weight_decay=WEIGHT_DECAY,
                                                 betas=(BETA1, BETA2))

    # Load data
    train_data, val_data = load_data()
    data = {'train': train_data, 'val': val_data}

    # Estimate number of epochs based on the parameters
    steps_per_epoch = len(train_data) / (BATCH_SIZE * model_config['block_size'])
    total_epochs = MAX_ITERS / steps_per_epoch
    print(f"[TRAINING INFO]: Estimated number of epochs: {total_epochs:.2f}")
    
    # Train model
    base_model_path = train_model(model, optimizer, data, model_config, out_dir, model_name)

    # Load best model and generate text
    load_best_model(model, optimizer, base_model_path)


if __name__ == "__main__":
    main()
