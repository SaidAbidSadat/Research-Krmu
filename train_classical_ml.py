import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time

# Load data
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# Handle NaN values in cleaned_text
train_df['cleaned_text'] = train_df['cleaned_text'].fillna('')
test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}\n")

# TF-IDF Vectorization
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
X_test = vectorizer.transform(test_df['cleaned_text'])
y_train = train_df['label']
y_test = test_df['label']

print(f"Feature matrix shape: {X_train.shape}\n")

# Dictionary to store results
results = {}

# 1. Logistic Regression
print("=" * 60)
print("Training Logistic Regression...")
start_time = time.time()
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_time = time.time() - start_time

lr_accuracy = accuracy_score(y_test, lr_pred)
results['Logistic Regression'] = {
    'accuracy': lr_accuracy,
    'training_time': lr_time,
    'predictions': lr_pred
}

print(f"Training Time: {lr_time:.2f} seconds")
print(f"Accuracy: {lr_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=['Not Cyberbullying', 'Cyberbullying']))

# 2. Support Vector Machine (SVM)
print("=" * 60)
print("Training Support Vector Machine...")
start_time = time.time()
svm_model = LinearSVC(random_state=42, max_iter=1000)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_time = time.time() - start_time

svm_accuracy = accuracy_score(y_test, svm_pred)
results['SVM'] = {
    'accuracy': svm_accuracy,
    'training_time': svm_time,
    'predictions': svm_pred
}

print(f"Training Time: {svm_time:.2f} seconds")
print(f"Accuracy: {svm_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, svm_pred, target_names=['Not Cyberbullying', 'Cyberbullying']))

# 3. Random Forest
print("=" * 60)
print("Training Random Forest...")
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_time = time.time() - start_time

rf_accuracy = accuracy_score(y_test, rf_pred)
results['Random Forest'] = {
    'accuracy': rf_accuracy,
    'training_time': rf_time,
    'predictions': rf_pred
}

print(f"Training Time: {rf_time:.2f} seconds")
print(f"Accuracy: {rf_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['Not Cyberbullying', 'Cyberbullying']))

# Summary comparison
print("=" * 60)
print("\nMODEL COMPARISON SUMMARY:")
print("-" * 60)
for model_name, metrics in results.items():
    print(f"{model_name:20s} | Accuracy: {metrics['accuracy']:.4f} | Time: {metrics['training_time']:.2f}s")

# Save best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\nBest Classical ML Model: {best_model_name}")

# Save models
pickle.dump(lr_model, open('logistic_regression_model.pkl', 'wb'))
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))
pickle.dump(rf_model, open('random_forest_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

print("\nAll models saved successfully!")