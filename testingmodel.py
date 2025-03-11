from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the saved model and tokenizer
model_path = "./final_sentiment_model"
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

def predict_sentiment(text):
    # Ensure model is in evaluation mode
    loaded_model.eval()
    
    # Tokenize input text
    inputs = loaded_tokenizer(
        text, 
        return_tensors="pt", 
        padding=True,
        truncation=True, 
        max_length=128
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
    
    # Map prediction to sentiment (reversed from your original mapping)
    sentiment_map = {
        1: "negative",  # Switched from 0
        0: "positive"   # Switched from 1
    }
    
    confidence = probabilities[0][prediction].item()
    return sentiment_map[prediction], confidence

if __name__ == "__main__":
    # Test cases
    test_cases = [
        "This product is BEST!",
        "I absolutely love this product!",
        "This is terrible, don't buy it",
        "Amazing product, highly recommend!",
        "Waste of money, very disappointed"
    ]
    
    print("Running sentiment analysis tests:\n")
    for text in test_cases:
        sentiment, confidence = predict_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 50 + "\n")

# Example usage
text = "This product is BEST!"
sentiment, confidence = predict_sentiment(text)
print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2f}")