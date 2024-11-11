'''
A module for shared functions between different training stages (AKA pretrain and fintune)
to allow forcode reuseability as much as possible.
'''

import os
import psutil
import torch
import json
from contextlib import nullcontext

# Import necessary shared components
from gpt import GPT

# Model Configuration Path
CONFIG_PATH = 'model_config.json'
DATA_DIR = 'data'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'float16' if torch.cuda.is_available() else 'float32'
DEVICE_TYPE = 'cuda' if 'cuda' in DEVICE else 'cpu'
PTDTYPE = {'float32': torch.float32, 'float16': torch.float16}[DTYPE]
CTX = nullcontext() if DEVICE_TYPE == 'cpu' else torch.amp.autocast(device_type=DEVICE_TYPE, dtype=PTDTYPE)

def load_configurations():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def cpu_memory_usage():
    memory_info = psutil.virtual_memory()
    usage_percentage = memory_info.percent
    return f"{usage_percentage:.2f}%"
    
    
def get_model_choice(configs):
    # Display available model configurations and prompt the user to choose
    print("Available model configurations:")
    for idx, model_name in enumerate(configs.keys()):
        print(f"{idx + 1}: {model_name}")
    choice = int(input("Select a model configuration by number: ")) - 1
    model_name = list(configs.keys())[choice]
    return model_name, configs[model_name]


def initialize_model(model_config, model_name, step='base_model', learning_rate=2e-4, weight_decay=0.01, betas=(0.9, 0.98)):
    """
    Initialize the model for training purposes, allowing you to continue training from a checkpoint or start from scratch.
    If a checkpoint is found, the optimizer state is restored and updated with the provided parameters.
    
    Args:
        model_config (dict): From model_config.json
        model_name (str): The key from model_config, the model's architecture to train.
        step (str): Choose from 'base_model' or 'fine_tuned'.
        learning_rate (float, optional): Learning rate for the optimizer. Value based on step.
        weight_decay (float, optional): Weight decay for the optimizer. Value based on step.
        betas (tuple, optional): (BETA1, BETA2) for the optimizer. Value based on step.
    """
    out_dir = f'out/{model_name}'
    os.makedirs(out_dir, exist_ok=True)

    # Set the appropriate checkpoint path based on the step
    checkpoint_path = os.path.join(out_dir, f"{model_name}_{step}.pt")
    model = GPT(model_config)

    # Configure a new optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, DEVICE)

    # Check if a checkpoint exists for the chosen step
    if os.path.exists(checkpoint_path):
        user_choice = input(f"[INFO]: Checkpoint found at {checkpoint_path}. Do you want to continue training from this checkpoint? (y/n): ").strip().lower()
        if user_choice == 'y':
            print(f"[INFO]: Loading weights and optimizer state from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Adjust optimizer parameters based on the current step
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                param_group['weight_decay'] = weight_decay
            optimizer.defaults['betas'] = betas
            print(f"[INFO]: Updated optimizer with lr={learning_rate}, weight_decay={weight_decay}, betas={betas}.")
        else:
            print("[INFO]: Starting training from scratch.")
    else:
        print(f"[INFO]: No checkpoint found for {step}. Starting training from scratch.")

    # Move model to the appropriate device
    model.to(DEVICE)
    return model, optimizer, out_dir


def load_model(model_config, model_name, step='fine_tuned'):
    """
    Load a trained model for inference.
    
    Args:
        model_config (dict): The configuration dictionary for the model.
        model_name (str): The name of the model directory.
        step (str): The training step to load ('fine_tuned' by default).
    
    Returns:
        tuple: (model, out_dir)
            - model: The loaded GPT model.
            - out_dir: The directory where the model is stored.
    """
    out_dir = f'out/{model_name}'
    os.makedirs(out_dir, exist_ok=True)

    # Set the checkpoint path
    checkpoint_path = os.path.join(out_dir, f"{model_name}_{step}.pt")

    # Initialize the model
    model = GPT(model_config)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR]: No checkpoint found at {checkpoint_path}. Cannot load the model.")

    print(f"[INFO]: Loading model weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to the appropriate device
    model.to(DEVICE)
    model.eval()  # Ensure model is in evaluation mode
    return model, out_dir