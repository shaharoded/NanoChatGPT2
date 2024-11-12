import os
import requests
import json
import tiktoken
import numpy as np
import gzip

# Initialize the tiktoken encoding object
TOKENIZER = tiktoken.get_encoding("gpt2")

# Tokenization functions
def encode(text, end_token=False):
    """
    Encode a string into a list of token IDs using GPT-2 BPE.
    Args:
        end_token (bool): Add the <|endoftext|> token to the encoding process (needed for QA)
    """
    if end_token:
        return TOKENIZER.encode_ordinary(text) + [TOKENIZER.eot_token]  # eot_token is the ID for <|endoftext|>
    return TOKENIZER.encode_ordinary(text)

def decode(token_ids):
    """Decode a list of token IDs back into a string using GPT-2 BPE."""
    return TOKENIZER.decode(token_ids)

def download_data(url, output_file_name=None):
    """
    Download a file from a given URL. 
    The file is either a .txt file or .txt.gz file, so extraction will be performed if needed.
    The function will get the .txt file and save locally under output_path.
    Return the concatenated text as str.
    Args:
        url (str): The URL to download from.
        output_file_name (str): The name to save the downloaded (extracted if needed) txt or json file in. Default to None, in which case no file is saved.
    Returns:
        str: Concatenated text from the downloaded or extracted files.
    """        
    # Get the file name from the URL and set the full path
    output_dir = os.path.dirname(__file__)
    original_file_name = url.split('/')[-1]
    original_file_path = os.path.join(output_dir, original_file_name)
    if output_file_name:
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
        if output_file_name:
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
        if output_file_name:
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
        if output_file_name:
            with open(saved_file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=4)
        
        os.remove(original_file_path)  # Remove the original file after saving
        return json.dumps(content, ensure_ascii=False, indent=4)  # Return the JSON content as a formatted JSON string
    else:
        raise ValueError(f'Unrecognized file type found, the original file downloaded is: {original_file_name}')
    
# Function to preprocess Q&A data
def preprocess_qa_data(data_str, add_context=False, output_file_name='qa_input.txt'):
    """
    Preprocess Q&A data from a JSON file and return a continuous stream of tokenized text,
    while also printing statistics on input and output lengths for QA pairs.
    
    Args:
        data_str (str): A formatted JSON string containing the QA data.
        add_context (bool): Add the paragraph context to the QA data.
        eot_token_id (int): Token ID for <EOT>. If not provided, defaults to the tokenizer's EOT token.
    
    Returns:
        np.array: A continuous stream of token IDs for all QA pairs, concatenated with <EOT>.
    """
    # set path to save .txt
    output_dir = os.path.dirname(__file__)    
    data = json.loads(data_str)
    tokenized_stream = []
    input_lengths = []  # Track lengths for statistics

    eot_token_id = TOKENIZER.eot_token

    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']  # Extract the context
            for qa in paragraph['qas']:
                question = qa['question']
                if not question.endswith("?"):
                    question += "?"  # Ensure question ends with a "?"

                for answer in qa['answers']:
                    answer = answer['text']
                    
                    # Mild formatting for answers
                    if answer.endswith('!'):
                        answer = answer[:-1] + '.'
                    elif not answer.endswith('.'):
                        answer += '.' 
                    
                    # Handle context
                    input_text = f"{context}\n{question}" if add_context else f"{question}"
                    output_text = f"{answer[:1].upper()}{answer[1:]}" # Properly formatting answer's as sentences
                    text_qa = input_text + ' ' + output_text + ' '

                    # Concatenate QA pair with EOT token
                    tokenized_qa = encode(text_qa, end_token=False) + [eot_token_id]
                    tokenized_stream.extend(tokenized_qa)

                    # Update length stats
                    input_lengths.append(len(tokenized_qa))

    # Print statistics
    print(f"[DATA STATS]: Input Lengths (tokens) -> Min: {min(input_lengths)}, Max: {max(input_lengths)}, Avg: {sum(input_lengths) / len(input_lengths):.2f}")
    
    # Save qa text in folder 
    if output_file_name:
        text_stream = decode(tokenized_stream)
        saved_file_path = os.path.join(output_dir, output_file_name)
        with open(saved_file_path, "w", encoding="utf-8") as f:
                f.write(text_stream)
        
    return np.array(tokenized_stream, dtype=np.int64)

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
    train_ids = encode(train_data, end_token=False)
    val_ids = encode(val_data, end_token=False)
    print(f"[RUNTIME INFO]: Pretrain train has {len(train_ids):,} tokens")
    print(f"[RUNTIME INFO]: Pretrain val has {len(val_ids):,} tokens")

    # Export train and validation to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.astype(np.int32).tofile(os.path.join(os.path.dirname(__file__), 'pretrain_train.bin'))
    val_ids.astype(np.int32).tofile(os.path.join(os.path.dirname(__file__), 'pretrain_val.bin'))
    
    # Preprocess QA Data
    qa_options = config["qa_data"]
    print("Available QA text corpora:")
    for idx, (name, url) in enumerate(qa_options.items()):
        print(f"{idx + 1}: {name}")

    choice = int(input("Select a QA corpus by number: ")) - 1
    selected_qa_name = list(qa_options.keys())[choice]
    selected_qa_url = qa_options[selected_qa_name]

    # Download QA data and turn to tokenized stream
    qa_input_path = os.path.join(os.path.dirname(__file__))
    qa_extracted_dict = download_data(selected_qa_url, qa_input_path)
    qa_tokenized_stream = preprocess_qa_data(qa_extracted_dict)
    
    # Define split point for train and validation
    split_index = int(len(qa_tokenized_stream) * 0.9)

    # Find the last <EOT> token in the training split
    while split_index < len(qa_tokenized_stream) and qa_tokenized_stream[split_index] != TOKENIZER.eot_token:
        split_index += 1

    # Include the <EOT> token in the training set and start validation after it
    qa_train_stream = qa_tokenized_stream[:split_index + 1]
    qa_val_stream = qa_tokenized_stream[split_index + 1:]

    print(f"[RUNTIME INFO]: <EOT> text in this tokenizer: {decode([TOKENIZER.eot_token])}")
    print(f"[RUNTIME INFO]: Data stream length is {len(qa_tokenized_stream)} tokens")
    print(f"[RUNTIME INFO]: Adjusted split point at token {split_index} to align with <EOT>")
    print(f"[RUNTIME INFO]: Train stream ends with '{decode([qa_train_stream[-1]])}' (should be <EOT>)")
    print(f"[RUNTIME INFO]: Validation stream starts with '{decode([qa_val_stream[0]])}' (should not be <EOT>)")

    # Save tokenized streams
    qa_train_stream.astype(np.int32).tofile(os.path.join(os.path.dirname(__file__), 'qa_train.bin'))
    qa_val_stream.astype(np.int32).tofile(os.path.join(os.path.dirname(__file__), 'qa_val.bin'))

    # Debug information
    print(f"[RUNTIME INFO]: QA train data saved with {len(qa_train_stream):,} tokens")
    print(f"[RUNTIME INFO]: QA val data saved with {len(qa_val_stream):,} tokens")

    print("[RUNTIME STATUS]: Data preprocessing and bin file creation complete.")