# Training a Nano Version of ChatGPT2
This project aims to train a language model from scratch, making it as functional and user-friendly as possible. It is an adaptation of `https://github.com/karpathy/nanoGPT/tree/master` with a focus on simplifying the process and adding basic "instruction-following" capabilities to the model.

DISCLAIMER - This is a reduced architecture on small datasets allowing training on CPU. This means the model performances will be pretty lousy. Bigger datasets (base + QA) on bigger architecture (as suggested in `model_config.json`) will significantly improve the model's quality.

## Getting Started
1. Clone this repository. Run the following command:

```bash
git clone https://github.com/shaharoded/NanoChatGPT2.git
cd NanoChatGPT2
```

2. Set up a virtual environment and activate it:

```bash
python -m venv venv
.\venv\Scripts\Activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Pre-Train the Model
This part focuses on training the initial text generation model based on chosen .txt.

### 1. Load Train and Validation Data
A first step would be to load, tokenize and encode the data. In the `data\data_config.json` file you'll find the link to access the designated training data. The `data_load.py` file holds the simple code to load the text and tokenize it based on the chosen tokenizer (hardcoded to match OpenAI's). Run the following (mind the user prompts):

```bash
cd data
python data_load.py
```
This code will prompt you to choose a base text to train on. The size of the text matters to the number of needed iterations and model complexity.
In order to later train the model to QA, opensource QA datasource was found and preprocessed here as well.
After running this file you'll have 6 new files in this repository:

1. `pretrain_input.txt` - The full readable .txt file - Your entire data
2. `pretrain_train.bin` - The tokenized train data for the model, holding 90% of the dataset.
3. `pretrain_val.bin` - The tokenized validation data for the model, holding 10% of the dataset.
4. `qa_input.txt` - The full .json file of the question / answer data, unprocessed.
5. `qa_train.bin` - The tokenized train data for the model, holding 90% of the dataset.
6. `qa_val.bin` - The tokenized validation data for the model, holding 10% of the dataset.


Those can later be accessed in the training loops. This step should only be performed once to generate the data files, as newer files will run over older files.

NOTE: The tokenizer is hardcoded in this module and imported to other connected files.

### 2. Train a Base Model Object
The base GPT model is an nn.Module model structured in the file `gpt.py`. After creating the datasets, this model can be trained using the `pretrain.py` module, to create a language model capable of generating text based on learnt corpus. This is the base GPT model, later fineuned into an assistant.
Using this process will create `out\{model_name}` directories with the model's best checkpoint. The best model will be saved under the name `{model_name}_base_model.pt`. This model will then be loaded to generate text as a POC. Using nanoGPT configurations with data of ~73MB on the current train configurations took about 8 hours (CPU).
Use the following to run (mind the user prompts):

```bash
python pretrain.py
```
NOTE: This module defines `torch.manual_seed(1337)` meaning every random process is the consistent. This also causes the trained model to generate the same responses to re-used test cases, which might contradict the expectation for some randomness in the responses.

### 3. Finetune the Base Model to QA
As described in step #1, we already loaded QA data for the finetune part. Now, this module focuses on taking this data and train the base model to the task of an assistant. activate
After the train is concluded, the model will generate response to a few hardcoded questions.
Finetuning the nanoGPT model with QA data of ~88MB on the current train configurations took about 4.5 hours (CPU).

Use the following to run (mind the user prompts):

```bash
python qa_finetune.py
```

### 4. Reinforcement Learning from Human Feedback (RLHF)
The final stage of training a conversational bot like this typically involves Reinforcement Learning from Human Feedback (RLHF). This stage helps align the model's behavior with human preferences, such as being more helpful, accurate, or less biased.

Currently, due to limitations in resources and time (lack of tagged responses and a reinforcement mechanism for feedback), this step is not implemented. However, feel free to extend the project and experiment with RLHF.

### 5. Try Your Model
This project also offers a `playground.py` module that allows you to insert free prompts to a trained model of your choice (from `out` repository, generated after training at least 1 model). This module is added with functionality to collect user's feedback which might later be used to fine-tune the model based on RLHF, as described in #4. The feedback data will be saved in data repository.

In order to start:

```bash
python playground.py
```

## TO-DO
1. Add 'continue training' functionality for both pretrain and finetune stage from existing model weights
2. Integrate memory track in the process
3. Fix finetune module

## GitHub Push Actions
To commit and push all changes to the repository follow these steps:

    ```bash
    git init
    git remote add origin https://github.com/shaharoded/NanoChatGPT2.git
    git fetch origin
    git add .
    git commit -m "Reasons for disrupting GIT (commit message)"
    git branch -M main
    git push -u origin main / git push -f origin main   # If you want to force push
    ```

    *Note: Replace `main` with your branch name if you're not using the `main` branch.*