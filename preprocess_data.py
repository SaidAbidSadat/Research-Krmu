import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Load dataset
df = pd.read_csv('cyberbullying_tweets.csv')

# Create binary labels: 1 = cyberbullying, 0 = not cyberbullying
df['label'] = df['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtag symbol but keep word
    return text.strip()

# Apply cleaning
df['cleaned_text'] = df['tweet_text'].apply(clean_text)

# Split data: 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Save processed data
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
print(f"\nTraining set label distribution:\n{train_df['label'].value_counts()}")
print(f"\nTest set label distribution:\n{test_df['label'].value_counts()}")
