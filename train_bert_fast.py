import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data and USE SUBSET for faster training
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# Use 30% of training data for faster training
train_df = train_df.sample(frac=0.3, random_state=42)
print(f"Using {len(train_df)} training samples (30% subset)")
print(f"Using {len(test_df)} test samples (full test set)")

# Custom Dataset class
class CyberbullyingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):  # Reduced max_len
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer and model
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = model.to(device)

# Create datasets
train_dataset = CyberbullyingDataset(
    train_df['cleaned_text'].values,
    train_df['label'].values,
    tokenizer
)

test_dataset = CyberbullyingDataset(
    test_df['cleaned_text'].values,
    test_df['label'].values,
    tokenizer
)

# Create dataloaders with larger batch size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Training setup - ONLY 1 EPOCH
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 1

print(f"\nStarting training for {epochs} epoch...")
print("Estimated time: 2-3 hours on CPU\n")

# Training loop
model.train()
total_loss = 0

for batch_idx, batch in enumerate(train_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    
    if (batch_idx + 1) % 50 == 0:
        print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

avg_loss = total_loss / len(train_loader)
print(f"\nTraining completed. Average Loss: {avg_loss:.4f}")

# Evaluation
print("\nEvaluating on test set...")
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=['Not Cyberbullying', 'Cyberbullying']))

# Save model
print("\nSaving model...")
model.save_pretrained('./cyberbullying_bert_model')
tokenizer.save_pretrained('./cyberbullying_bert_model')
print("BERT model saved successfully!")