import torch
import os
import json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

# Local code
from data.data_load import encode
from pretrain import (
    DEVICE,
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
    return answer

if __name__ == "__main__":
    print("Oi, I'm listenin'. Ask your bloody question, or type 'quit' to sod off.")
    
    output_dir = os.path.join("data", f"{model_name}_feedback_data")
    feedback_file = os.path.join(output_dir, "feedback_data.json")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load existing feedback if the file already exists
    feedback_data = []
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)

    while True:
        prompt = input("What the f*** do you want? ")

        if prompt.lower() in ['quit', 'exit', 'leave']:
            print("Alright, off ya go then. Bugger off.")
            break
        else:
            # Generate response from model
            response = generate_response(model, prompt)
            print(f"Model's Response: {response}")

            # Ask the user for a rating and additional feedback
            try:
                rating = int(input("Rate the response from 1 (absolute sh**e) to 5 (brilliant): "))
                if rating < 1 or rating > 5:
                    raise ValueError("Oi, follow instructions! Enter a bloody number between 1 and 5.")
                if rating < 3:
                    print("You annoying wanker... I know it ain't perfect, but cut me some slack, yeah?")
            except ValueError:
                continue

            additional_feedback = input("Any other thoughts? (Press Enter to skip): ")

            # Collect feedback
            feedback = {
                "prompt": prompt,
                "response": response,
                "rating": rating,
                "additional_feedback": additional_feedback
            }

            # Append feedback to list
            feedback_data.append(feedback)

            # Save feedback data to JSON file
            with open(feedback_file, "w") as f:
                json.dump(feedback_data, f, indent=4)
            
            print('\nOK. Next!')
        