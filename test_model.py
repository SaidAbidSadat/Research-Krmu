import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import re

print("="*60)
print("CYBERBULLYING DETECTION SYSTEM")
print("="*60)

# Load models
print("\nLoading models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load BERT
bert_tokenizer = BertTokenizer.from_pretrained('./cyberbullying_bert_model')
bert_model = BertForSequenceClassification.from_pretrained('./cyberbullying_bert_model')
bert_model.to(device)
bert_model.eval()

# Load Classical ML models
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lr_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

print("✓ Models loaded successfully!\n")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    return text.strip()

# BERT prediction function
def predict_bert(text):
    cleaned = clean_text(text)
    encoding = bert_tokenizer.encode_plus(
        cleaned,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
    
    return prediction, probabilities.cpu().numpy()

# Classical ML prediction function
def predict_classical(text):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = lr_model.predict(features)[0]
    probabilities = lr_model.predict_proba(features)[0]
    return prediction, probabilities

# Test examples
test_examples = [
    "You're such an idiot, no one likes you!",
    "Great job on your presentation today!",
    "Go back to your country, we don't want you here",
    "Love your new profile picture!",
    "You're too old to understand technology"
]

print("TESTING WITH SAMPLE TEXTS:")
print("-"*60)

for i, text in enumerate(test_examples, 1):
    print(f"\n{i}. Text: \"{text}\"")
    
    # BERT prediction
    bert_pred, bert_probs = predict_bert(text)
    bert_label = "CYBERBULLYING" if bert_pred == 1 else "NOT CYBERBULLYING"
    bert_confidence = bert_probs[bert_pred] * 100
    
    # Classical ML prediction
    lr_pred, lr_probs = predict_classical(text)
    lr_label = "CYBERBULLYING" if lr_pred == 1 else "NOT CYBERBULLYING"
    lr_confidence = lr_probs[lr_pred] * 100
    
    print(f"   BERT:      {bert_label} (confidence: {bert_confidence:.2f}%)")
    print(f"   Log. Reg.: {lr_label} (confidence: {lr_confidence:.2f}%)")

print("\n" + "="*60)
print("INTERACTIVE MODE - Test your own text!")
print("(Type 'quit' to exit)")
print("="*60)

while True:
    user_input = input("\nEnter text to analyze: ").strip()
    
    if user_input.lower() == 'quit':
        print("\nThank you for using the Cyberbullying Detection System!")
        break
    
    if not user_input:
        print("Please enter some text.")
        continue
    
    # Predictions
    bert_pred, bert_probs = predict_bert(user_input)
    lr_pred, lr_probs = predict_classical(user_input)
    
    bert_label = "🚨 CYBERBULLYING" if bert_pred == 1 else "✅ NOT CYBERBULLYING"
    lr_label = "🚨 CYBERBULLYING" if lr_pred == 1 else "✅ NOT CYBERBULLYING"
    
    print(f"\n   BERT Model:        {bert_label} (confidence: {bert_probs[bert_pred]*100:.2f}%)")
    print(f"   Logistic Reg.:     {lr_label} (confidence: {lr_probs[lr_pred]*100:.2f}%)")