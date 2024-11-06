import torch
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

# Local code
from data.data_load import encode
from pretrain import (
    BETA1, BETA2, DEVICE, WEIGHT_DECAY, LEARNING_RATE, DEVICE_TYPE,
    load_configurations, get_model_choice, initialize_model
)

print("Oi, welcome to the bloody Play Ground. Pick a trained model, and make it quick, ya wanker...")
# Directory where base models are saved
OUT_DIR = "out"
CONFIG_PATH = 'model_config.json'

# Load configurations and initialize model
model_directories = [
d for d in os.listdir(OUT_DIR)
if os.path.isdir(os.path.join(OUT_DIR, d)) and
any(f.endswith('fine_tuned.pt') for f in os.listdir(os.path.join(OUT_DIR, d)))
]
configs = load_configurations()

# Filter configurations that match the model directories
configs = {model_name: config for model_name, config in configs.items() if model_name in model_directories}
if len(configs) == 0:
    raise Exception("You got nothin', mate. No trained models at all. Bloody pathetic...")

# Get model choice from the available configurations
model_name, model_config = get_model_choice(configs)  # This returns model_name and model_config correctly

# Initialize model (overrun OUT_DIR with relevant OUT_DIR)
model, OUT_DIR = initialize_model(model_config, model_name)
model_path = os.path.join(OUT_DIR, f'{model_name}_fine_tuned.pt')
optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2), DEVICE_TYPE)

checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)

def generate_response(model, prompt):
    """
    Generate response to a given prompt.
    Args:
        model (GPT): A pretrained finetuned class object used to generate response. 
                    Loaded before running this function.
        prompt (str): Initial prompt to the model to generate on top.
    """
    model.eval()  # Ensure model is in evaluation mode
    input_tokens = encode(prompt)
    input_tensor = torch.tensor(input_tokens, dtype=torch.int64).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        answer = model.generate_text(input_tensor, max_new_tokens=200, device=DEVICE)
    # Remove the prompt from the generated response
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    return f"Model's Response: {answer}"

if __name__ == "__main__":
    print("Oi, I'm listenin'. Ask your bloody question, or type 'quit' to sod off.")
    
    while True:
        prompt = input("What the f*** do you want? ")
        
        if prompt.lower() in ['quit', 'exit', 'leave']:
            print("Alright, off ya go then. Bugger off.")
            break
        else:
            print(generate_response(model, prompt))
        