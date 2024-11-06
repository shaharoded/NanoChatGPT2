import os
import requests
import json
import tiktoken
import numpy as np
import gzip

# Initialize the tiktoken encoding object
tokenizer = tiktoken.get_encoding("gpt2")

# Tokenization functions
def encode(text, end_token=False):
    """
    Encode a string into a list of token IDs using GPT-2 BPE.
    Args:
        end_token (bool): Add the <|endoftext|> token to the encoding process (needed for QA)
    """
    if end_token:
        return tokenizer.encode_ordinary(text) + [tokenizer.eot_token]  # eot_token is the ID for <|endoftext|>
    return tokenizer.encode_ordinary(text)

def decode(token_ids):
    """Decode a list of token IDs back into a string using GPT-2 BPE."""
    return tokenizer.decode(token_ids)

def download_data(url, output_file_name):
    """
    Download a file from a given URL. 
    The file is either a .txt file or .txt.gz file, so extraction will be performed if needed.
    The function will get the .txt file and save locally under output_path.
    Return the concatenated text as str.
    Args:
        url (str): The URL to download from.
        output_file_name (str): The name to save the downloaded (extracted if needed) txt or json file in.
    Returns:
        str: Concatenated text from the downloaded or extracted files.
    """        
    # Get the file name from the URL and set the full path
    output_dir = os.path.dirname(__file__)
    original_file_name = url.split('/')[-1]
    original_file_path = os.path.join(output_dir, original_file_name)
    saved_file_path = os.path.join(output_dir, output_file_name)

    # Download the file
    print(f"[RUNTIME STATUS]: Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the download fails
    with open(original_file_path, "wb") as f:
        f.write(response.content)

    # Check file type and process accordingly
    if original_file_name.endswith(".gz"):
        print("[RUNTIME STATUS]: Extracting the gz file...")
        with gzip.open(original_file_path, "rt", encoding="utf-8", errors="ignore") as f:
            extracted_text = f.read()
        
        # Save the extracted text to saved_file_path
        with open(saved_file_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        os.remove(original_file_path)  # Remove the gz file after extraction
        return extracted_text

    elif original_file_name.endswith(".txt"):
        # If it's a plain text file, read and save the content
        print("[RUNTIME STATUS]: Handling plain txt file...")
        with open(original_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Save the content to saved_file_path
        with open(saved_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        os.remove(original_file_path)  # Remove the original file after saving
        return content

    elif original_file_name.endswith(".json"):
        # If it's a JSON file, read and save the content
        print("[RUNTIME STATUS]: Handling JSON file...")
        with open(original_file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # Save the content to saved_file_path in a pretty JSON format
        with open(saved_file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
        
        os.remove(original_file_path)  # Remove the original file after saving
        return json.dumps(content, ensure_ascii=False, indent=4)  # Return the JSON content as a formatted JSON string
    else:
        raise ValueError(f'Unrecognized file type found, the original file downloaded is: {original_file_name}')
    
# Function to preprocess Q&A data
def preprocess_qa_data(data_str):
    """
    Preprocess Q&A data from a JSON file and return formatted question-answer pairs.
    Input is a formatted json str derived from the json in the previous function.
    """
    data = json.loads(data_str)
    qa_pairs = []

    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    answer_text = answer['text']
                    # Format: "Q: [question] A: [answer] <|endoftext|>"
                    qa_pair = f"Q: {question} A: {answer_text}"
                    encoded_qa_pair = encode(qa_pair, end_token=True)
                    qa_pairs.append(encoded_qa_pair)
    return qa_pairs

# Code to preprocess the data, Will only run if the module is called directly
if __name__ == "__main__":
    # Reading JSON data
    config_path = os.path.join(os.path.dirname(__file__), 'data_config.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Display available corpora options
    corpora_options = config["train_corpora"]
    print("Available text corpora:")
    for idx, (name, url) in enumerate(corpora_options.items()):
        print(f"{idx + 1}: {name}")

    # Prompt user to select a text corpus
    choice = int(input("Select a text corpus by number: ")) - 1
    selected_corpus_name = list(corpora_options.keys())[choice]
    selected_corpus_url = corpora_options[selected_corpus_name]

    # Download pretraining data
    pretrain_output_path = os.path.join(os.path.dirname(__file__), 'pretrain_input.txt')
    pretrain_extracted_text = download_data(selected_corpus_url, pretrain_output_path)
        
    n = len(pretrain_extracted_text)
    size_kb = n / 1024  # Size in kilobytes
    print(f"[RUNTIME INFO]: Approximate size of the loaded text: {size_kb:.2f} KB ({(size_kb/1024):.2f} MB).")
    train_data = pretrain_extracted_text[:int(n*0.9)]
    val_data = pretrain_extracted_text[int(n*0.9):]

    # Encode train and validation data
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"[RUNTIME INFO]: Pretrain train has {len(train_ids):,} tokens")
    print(f"[RUNTIME INFO]: Pretrain val has {len(val_ids):,} tokens")

    # Export train and validation to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'pretrain_train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'pretrain_val.bin'))
    
    # Preprocess QA Data
    qa_options = config["qa_data"]
    print("Available QA text corpora:")
    for idx, (name, url) in enumerate(qa_options.items()):
        print(f"{idx + 1}: {name}")

    choice = int(input("Select a QA corpus by number: ")) - 1
    selected_qa_name = list(qa_options.keys())[choice]
    selected_qa_url = qa_options[selected_qa_name]

    # Download QA data
    qa_input_path = os.path.join(os.path.dirname(__file__), 'qa_input.json')
    qa_extracted_dict = download_data(selected_qa_url, qa_input_path)

    # Preprocess and split QA data
    qa_pairs = preprocess_qa_data(qa_extracted_dict)
    split_index = int(len(qa_pairs) * 0.9)
    qa_train_ids = qa_pairs[:split_index]
    qa_val_ids = qa_pairs[split_index:]

    # Flatten and save QA data
    qa_train_ids = np.array([token_id for pair in qa_train_ids for token_id in pair], dtype=np.uint16)
    qa_val_ids = np.array([token_id for pair in qa_val_ids for token_id in pair], dtype=np.uint16)
    print(f"[RUNTIME INFO]: QA train has {len(qa_train_ids):,} tokens")
    print(f"[RUNTIME INFO]: QA val has {len(qa_val_ids):,} tokens")

    qa_train_ids.tofile(os.path.join(os.path.dirname(__file__), 'qa_train.bin'))
    qa_val_ids.tofile(os.path.join(os.path.dirname(__file__), 'qa_val.bin'))

    print("[RUNTIME STATUS]: Data preprocessing complete.")