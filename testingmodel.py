#testingmodel.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import pandas as pd

# 1) Config
MODEL_NAME = "distilbert-base-uncased"  # Changed to match the trained model
ADAPTER_PATH = "./final_model"

# 2) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 3) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4) Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
)
base_model.config.pad_token_id = tokenizer.pad_token_id

# 5) Wrap with LoRA adapters
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()
    print("Model and adapters loaded successfully.")
except ValueError as e:
    print(f"Error loading adapters: {e}. Ensure the adapter was trained with a compatible model.")
    print("Falling back to base model without adapters.")
    model = base_model.to(device).eval()

# 6) Label map
id2label = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text: str):
    """Predict sentiment for a single review."""
    if not text or not text.strip():
        return "neutral", 0.0  # Default for empty input
    enc = tokenizer(
        text.strip(),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        if torch.isnan(logits).any():
            raise ValueError("Model produced NaN logits. Ensure the model is properly trained and saved.")
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        pred = int(torch.argmax(logits, dim=-1))
        confidence = probs[pred] if probs[pred] == probs[pred] else 0.0  # Handle NaN
    return id2label[pred], confidence

def predict_batch_sentiments(reviews: list):
    """Predict sentiments for a list of reviews."""
    if not reviews or not any(reviews):
        return [("neutral", 0.0) for _ in reviews]
    valid_reviews = [r.strip() for r in reviews if r and r.strip()]
    if not valid_reviews:
        return [("neutral", 0.0) for _ in reviews]
    inputs = tokenizer(
        valid_reviews,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        if torch.isnan(logits).any():
            raise ValueError("Model produced NaN logits. Ensure the model is properly trained and saved.")
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
    # Pad results to match input length
    result = [(id2label[p], probs[i].max() if probs[i].max() == probs[i].max() else 0.0) for i, p in enumerate(preds)]
    return result + [("neutral", 0.0)] * (len(reviews) - len(valid_reviews))

if __name__ == "__main__":
    while True:
        review = input("Enter a customer review (or 'quit' to exit): ")
        if review.lower() == 'quit':
            break
        if review.strip():
            label, confidence = predict_sentiment(review)
            print(f"Predicted Sentiment: {label} (confidence {confidence:.2f})")
        else:
            print("Please enter a non-empty review.")
