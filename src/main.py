from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import torch
import os

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Read the txt file and tokenize the document
def read_and_tokenize_book(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        text = file.read()
    tokens = tokenizer.encode(text)
    return tokens

# Function to split tokens into chunks
def chunk_tokens(tokens, chunk_size=1024):
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

class TextDataset(Dataset):
    def __init__(self, input_sequences, output_sequences):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_sequences[idx], dtype=torch.long),
            'labels': torch.tensor(self.output_sequences[idx], dtype=torch.long)
        }

def tokenizer_func(book_dir):
    input_sequences = []
    output_sequences = []
    
    for book_file in os.listdir(book_dir):
        if book_file.endswith(".txt"):
            file_path = os.path.join(book_dir, book_file)
            tokens = read_and_tokenize_book(file_path)
            token_chunks = chunk_tokens(tokens)
            
            for chunk in token_chunks:
                input_seq = chunk[:-1]
                output_seq = chunk[1:]
                input_sequences.append(input_seq)
                output_sequences.append(output_seq)

    # Create a dataset and data loader
    dataset = TextDataset(input_sequences, output_sequences)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=data_collator)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

# Ensure the path is correct and exists
book_directory = "E:\My Works\LLM model\src\cleaned_books"
if os.path.exists(book_directory):
    tokenizer_func(book_directory)
else:
    print(f"The directory {book_directory} does not exist.")
