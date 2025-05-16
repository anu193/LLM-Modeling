#model.py
import torch
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Optional: Try to import BitsAndBytesConfig
try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("Warning: bitsandbytes not installed. Install with `pip install -U bitsandbytes` for 8-bit quantization, or proceed without quantization.")

# 1) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2) Config
MODEL_NAME    = "distilbert-base-uncased"  # Changed to a model suited for sequence classification
PROCESSED_CSV = "processed_data_large.csv"
NUM_LABELS    = 3
LABEL2ID      = {"negative": 0, "neutral": 1, "positive": 2}

# 3) Load & prepare data
try:
    df = pd.read_csv(PROCESSED_CSV).dropna(subset=["reviewText", "sentiment"])
    if df.empty or len(df) < 10:
        raise ValueError("Processed data is empty or too small. Ensure processed_data_large.csv has sufficient data.")
    if not all(df["sentiment"].isin(LABEL2ID.keys())):
        raise ValueError("Invalid sentiment labels in processed_data_large.csv. Expected: negative, neutral, positive.")
    print("Data loaded successfully. Sentiment distribution:")
    print(df["sentiment"].value_counts())
    print(f"Sample reviewText entries: {df['reviewText'].head().tolist()}")
except FileNotFoundError:
    print(f"Error: '{PROCESSED_CSV}' not found. Run app.py first to generate processed_data_large.csv.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {str(e)}")
    sys.exit(1)

df["label"] = df["sentiment"].map(LABEL2ID)
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")

# Reset index & keep only the two columns
train_df = train_df[["reviewText", "label"]].reset_index(drop=True)
eval_df  = eval_df[["reviewText", "label"]].reset_index(drop=True)

# 4) Tokenizer & base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Configure quantization if available
bnb_config = None
if BNB_AVAILABLE and torch.cuda.is_available():
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        print("Using 8-bit quantization with bitsandbytes.")
    except Exception as e:
        print(f"Warning: Failed to configure bitsandbytes quantization: {str(e)}. Falling back to full precision.")
        bnb_config = None
else:
    print("No quantization: bitsandbytes not available or no CUDA device.")

try:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        torch_dtype=torch.float32,  # Use float32 to avoid numerical instability
        quantization_config=bnb_config,
    )
except Exception as e:
    print(f"Error loading model: {str(e)}. Ensure bitsandbytes is installed and MODEL_NAME is correct.")
    sys.exit(1)

# 5) PEFT/LoRA wrap for efficient fine-tuning
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],  # Adjusted for DistilBERT architecture
    lora_dropout=0.05,
    task_type="SEQ_CLS",
)
model = get_peft_model(base_model, peft_config).to(device)

# 6) Tokenization helper
def tokenize_batch(examples):
    # Ensure reviewText is non-empty and align labels
    reviews = [str(r).strip() for r in examples["reviewText"] if str(r).strip()]
    if not reviews:
        raise ValueError("No valid reviews to tokenize.")
    toks = tokenizer(
        reviews,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    # Filter labels to match the number of valid reviews
    labels = [label for review, label in zip(examples["reviewText"], examples["label"]) if str(review).strip()]
    toks["labels"] = labels
    return toks

# 7) Build datasets
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
eval_ds  = Dataset.from_pandas(eval_df, preserve_index=False)

try:
    train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["reviewText"])
    eval_ds  = eval_ds.map(tokenize_batch, batched=True, remove_columns=["reviewText"])
except Exception as e:
    print(f"Error tokenizing data: {str(e)}")
    sys.exit(1)

# 8) Metrics for business strategy evaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1_weighted": f1_score(p.label_ids, preds, average="weighted"),
    }

# 9) Training arguments
training_args = TrainingArguments(
    output_dir="distilbert_sentiment",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=3e-5,  # Slightly increased for better convergence
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=10,
)

# 10) Trainer with early stopping, gradient clipping, and NaN handling
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invalid_loss_count = 0
        self.max_invalid_loss = 5  # Stop after 5 consecutive invalid losses

    def training_step(self, model, inputs, *args, **kwargs):
        # Check for NaN in inputs
        for key, value in inputs.items():
            if torch.is_tensor(value) and torch.isnan(value).any():
                raise ValueError(f"NaN detected in inputs: {key}")
        step = super().training_step(model, inputs, *args, **kwargs)
        # Check for NaN in gradients
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                raise ValueError("NaN detected in gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return step

    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            if logs["loss"] != logs["loss"] or logs["loss"] == 0.0:
                print(f"Warning: Loss is invalid (NaN or 0) at step {self.state.global_step}: {logs['loss']}")
                self.invalid_loss_count += 1
                if self.invalid_loss_count >= self.max_invalid_loss:
                    raise ValueError("Training stopped: Too many consecutive invalid losses.")
            else:
                self.invalid_loss_count = 0  # Reset counter if loss is valid
        super().log(logs, *args, **kwargs)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# 11) Train & evaluate
try:
    trainer.train()
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)
except Exception as e:
    print(f"Error during training: {str(e)}")
    sys.exit(1)

# 12) Validate model before saving
try:
    model.eval()
    test_input = tokenizer(
        "great quality",
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**test_input).logits
    if torch.isnan(logits).any():
        raise ValueError("Model produces NaN logits after training. Training failed.")
    print("Model validation successful. Logits for test input:", logits.tolist())
except Exception as e:
    print(f"Error validating model: {str(e)}")
    sys.exit(1)

# 13) Save model for business strategy inference
try:
    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("✅ Sentiment model fine-tuned and saved to ./final_model. Training successful")
except Exception as e:
    print(f"Error saving model: {str(e)}")
    sys.exit(1)
