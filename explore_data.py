import pandas as pd

# Load the dataset
df = pd.read_csv('cyberbullying_tweets.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nClass Distribution:")
print(df['cyberbullying_type'].value_counts())
print("\nMissing Values:")
print(df.isnull().sum())