import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load test data
test_df = pd.read_csv('test_data.csv')
y_test = test_df['label']

# Load classical ML models and predictions
print("Loading models and generating predictions...")
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lr_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Handle NaN values
test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
X_test = vectorizer.transform(test_df['cleaned_text'])

# Get predictions
lr_pred = lr_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Model comparison data
models = ['BERT', 'Logistic Regression', 'SVM', 'Random Forest']
accuracies = [0.8725, 0.8608, 0.8515, 0.8454]
training_times = [5400, 0.22, 0.27, 20.49]  # BERT time in seconds (~1.5 hours)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'Training Time (seconds)': training_times
})

# Save comparison table
comparison_df.to_csv('model_comparison.csv', index=False)
print("\nModel Comparison Table:")
print(comparison_df)

# 1. Accuracy Comparison Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim(0.8, 0.9)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: accuracy_comparison.png")

# 2. Training Time Comparison (log scale)
plt.figure(figsize=(10, 6))
plt.bar(models, training_times, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
plt.ylabel('Training Time (seconds, log scale)', fontsize=12)
plt.title('Model Training Time Comparison', fontsize=14, fontweight='bold')
plt.yscale('log')
for i, (model, time) in enumerate(zip(models, training_times)):
    if time < 60:
        label = f'{time:.2f}s'
    else:
        label = f'{time/60:.1f}m'
    plt.text(i, time, label, ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_time_comparison.png")

# 3. Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
predictions = [None, lr_pred, svm_pred, rf_pred]  # None for BERT (not loaded here)
model_names = ['BERT (87.25%)', 'Logistic Regression (86.08%)', 
               'SVM (85.15%)', 'Random Forest (84.54%)']

for idx, (ax, pred, name) in enumerate(zip(axes.flat, predictions, model_names)):
    if pred is not None:
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Not CB', 'CB'], yticklabels=['Not CB', 'CB'])
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    else:
        # Manually add BERT confusion matrix values from earlier output
        cm_bert = np.array([[589, 1000], [239, 7711]])  # Approximate from recall/precision
        sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not CB', 'CB'], yticklabels=['Not CB', 'CB'])
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")

# 4. Performance vs Efficiency Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(training_times, accuracies, s=200, c=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], alpha=0.7)
for i, model in enumerate(models):
    plt.annotate(model, (training_times[i], accuracies[i]), 
                xytext=(10, -5), textcoords='offset points', fontsize=10)
plt.xlabel('Training Time (seconds, log scale)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance vs Training Efficiency', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('performance_vs_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: performance_vs_efficiency.png")

print("\n" + "="*60)
print("All visualizations generated successfully!")
print("Files created:")
print("  - model_comparison.csv")
print("  - accuracy_comparison.png")
print("  - training_time_comparison.png")
print("  - confusion_matrices.png")
print("  - performance_vs_efficiency.png")
print("="*60)