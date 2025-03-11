import pandas as pd

# Load your training data
train_data = pd.read_csv('train.csv')

# Check sentiment distribution
print("Sentiment distribution in training data:")
print(train_data['sentiment'].value_counts())

# Check unique sentiment labels
print("\nUnique sentiment labels:")
print(train_data['sentiment'].unique())

# Check a few examples
print("\nSample reviews and their sentiments:")
print(train_data[['reviewText', 'sentiment']].head())