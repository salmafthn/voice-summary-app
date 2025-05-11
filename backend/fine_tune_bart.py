from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk
import torch

# 1. Load the dataset
dataset = load_from_disk("D:/voice-summary-app/data/dataset")

# 2. Load the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# 3. Define device for GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 4. Tokenize the inputs and labels
def tokenize_function(examples):
    inputs = tokenizer(examples['content'], max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(examples['summary'], max_length=150, truncation=True, padding="max_length")
    inputs['labels'] = labels['input_ids']
    return inputs

# 5. Use smaller batches to reduce memory usage and processing time
tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=8)  # Using smaller batch size (8)

train_subset = dataset["train"].select(range(5000))  
test_subset = dataset["test"].select(range(5000))    

# Tokenize the subsets
tokenized_train_subset = train_subset.map(tokenize_function, batched=True, batch_size=8)
tokenized_test_subset = test_subset.map(tokenize_function, batched=True, batch_size=8)

# 7. Set up training arguments with gradient accumulation for smaller batch sizes
training_args = TrainingArguments(
    output_dir="./models",            # Output directory for the model checkpoints
    eval_steps=None,                   # Disabling evaluation steps or using `eval_steps` instead of `evaluation_strategy`
    logging_dir="./logs",              # Directory for storing logs
    logging_steps=100,                 # Log every 100 steps
    save_steps=500,                    # Save model checkpoint every 500 steps
    learning_rate=5e-5,                # Learning rate
    per_device_train_batch_size=2,     # Smaller batch size per device for training
    per_device_eval_batch_size=4,      # Smaller batch size per device for evaluation
    num_train_epochs=3,                # Number of training epochs
    weight_decay=0.01,                 # Weight decay for regularization
    gradient_accumulation_steps=16,    # Gradient accumulation steps to simulate larger batch size
)

# 8. Initialize the Trainer with the fine-tuning arguments and datasets
trainer = Trainer(
    model=model,                      # The model to train
    args=training_args,               # Training arguments
    train_dataset=tokenized_train_subset,  # Training dataset (subset)
    eval_dataset=tokenized_test_subset,   # Evaluation dataset (subset)
)

# 9. Fine-tune the model
trainer.train()

# 10. Save the fine-tuned model
trainer.save_model("./fine_tuned_bart")
