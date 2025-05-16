import pandas as pd
import re

INPUT_CSV = "amazon.csv"
OUTPUT_CSV = "processed_data_large.csv"

def map_sentiment(r):
    """Map numeric rating (as string or number) to a three-class sentiment."""
    try:
        r = float(r)
    except (ValueError, TypeError):
        return "neutral"
    if r >= 4.0:
        return "positive"
    elif r <= 2.0:
        return "negative"
    else:
        return "neutral"

def clean_text(text):
    """Clean text by removing URLs, special characters, and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove special characters
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

chunksize = 100_000
first = True

try:
    # Read only the two columns we care about, in chunks
    for chunk in pd.read_csv(
        INPUT_CSV,
        usecols=["review_content", "rating"],
        chunksize=chunksize,
        encoding='utf-8',
        on_bad_lines='skip'
    ):
        # Rename to match model.py expectations
        chunk = chunk.rename(columns={"review_content": "reviewText"})
        # Clean text
        chunk["reviewText"] = chunk["reviewText"].apply(clean_text).fillna("").str.strip()
        # Derive sentiment
        chunk["sentiment"] = chunk["rating"].apply(map_sentiment)
        # Drop the rating column after use
        chunk = chunk.drop(columns=["rating"])
        # Drop exact duplicates
        chunk = chunk.drop_duplicates(subset=["reviewText", "sentiment"])
        # Keep only the two columns
        out = chunk[["reviewText", "sentiment"]]

        # Write header on first chunk, then append
        if first:
            out.to_csv(OUTPUT_CSV, index=False, mode="w")
            first = False
        else:
            out.to_csv(OUTPUT_CSV, index=False, mode="a", header=False)

    print(f"✅ Finished processing. '{OUTPUT_CSV}' is ready for modeling.")

except FileNotFoundError:
    print(f"Error: '{INPUT_CSV}' not found.")
except Exception as e:
    print(f"Error during processing: {str(e)}")
