# Import required libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch  # Added for tensor type conversion

# Load and prepare the dataset
processed_file = 'processed_data_large.csv'
try:
    sentiment_data = pd.read_csv(processed_file)
    print(f"Loaded {processed_file} successfully.")
except FileNotFoundError:
    print(f"Error: '{processed_file}' not found. Ensure you run 'app.py' first.")
    exit()

# Convert text labels to numbers
if sentiment_data['sentiment'].dtype == 'object':
    le = LabelEncoder()
    sentiment_data['sentiment'] = le.fit_transform(sentiment_data['sentiment'])

# Split into training (90%) and evaluation (10%) sets
train_df, eval_df = train_test_split(sentiment_data, test_size=0.1, random_state=42)
print("Training data shape:", train_df.shape)
print("Evaluation data shape:", eval_df.shape)

# Initialize model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(sentiment_data['sentiment'].unique())
)

# Function to process text into model-readable format
def tokenize_and_format(examples):
    tokenized = tokenizer(
        examples["reviewText"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    # Convert labels to float tensor
    tokenized["labels"] = torch.tensor(examples["sentiment"], dtype=torch.float)
    return tokenized

# Convert pandas DataFrames to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Apply tokenization to both datasets
train_tokenized = train_dataset.map(
    tokenize_and_format, 
    batched=True, 
    remove_columns=train_dataset.column_names
)
eval_tokenized = eval_dataset.map(
    tokenize_and_format, 
    batched=True, 
    remove_columns=eval_dataset.column_names
)

# Configure training parameters
training_args = TrainingArguments(
    output_dir="./sentiment_model",        # Where to save model
    eval_strategy="epoch",                 # When to evaluate
    save_strategy="epoch",                 # When to save
    learning_rate=2e-5,                    # Learning rate
    per_device_train_batch_size=8,         # Batch size for training
    per_device_eval_batch_size=8,          # Batch size for evaluation
    num_train_epochs=3,                    # Number of training epochs
    weight_decay=0.01,                     # Weight decay for regularization
    logging_dir="./logs",                  # Directory for logs
    load_best_model_at_end=True,           # Save best model
    metric_for_best_model="loss",          # Metric to determine best model
    logging_steps=500                      # How often to log
)

# Function to compute accuracy metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    compute_metrics=compute_metrics
)

# Start training
print("Training started...")
try:
    trainer.train()
    print("Training completed successfully!")
    
    # Save the final model
    model.save_pretrained("./final_sentiment_model")
    tokenizer.save_pretrained("./final_sentiment_model")
    print("Model saved to ./final_sentiment_model")
    
except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    import psutil
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")