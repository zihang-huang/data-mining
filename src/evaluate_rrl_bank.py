import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from rrl.models import RRL
from rrl.utils import DBEncoder

# --- Configuration ---
DATA_PATH = 'datasets/bank-marketing/bank-full.csv'
MODEL_PATH = 'outputs/models/rrl_bank_model.pth'
PLOT_DIR = 'outputs/plots'
BATCH_SIZE = 64

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
except:
    pass
print(f"Using device: {device}")

# --- 1. Load and Preprocess Data (Must match training exactly) ---
print(f"Loading data from {DATA_PATH}...")
try:
    df = pd.read_csv(DATA_PATH, sep=';')
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    sys.exit(1)

print("Preprocessing data and re-fitting encoder...")
f_list = []
for col in df.columns[:-1]:
    if df[col].dtype == 'object':
        f_list.append([col, 'discrete'])
    else:
        f_list.append([col, 'continuous'])

f_df = pd.DataFrame(f_list)

# Initialize and Fit DBEncoder
# Important: We must fit on the original full dataset to ensure mappings (one-hot, scaling) match training.
db_enc = DBEncoder(f_df, discrete=False)
X_df = df.iloc[:, :-1]
y_df = df.iloc[:, -1:]
db_enc.fit(X_df, y_df)

X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

# Split data (use same random_state as training to get the same test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Test DataLoader
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. Load Model ---
print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
saved_args = checkpoint['rrl_args']
model_state_dict = checkpoint['model_state_dict']

# Re-instantiate RRL model with saved arguments
# Note: RRL __init__ arguments might need to be mapped if 'rrl_args' dict keys match __init__ params exactly.
# Based on rrl/models.py:
# saved_args keys: dim_list, use_not, use_skip, estimated_grad, use_nlaf, alpha, beta, gamma

rrl = RRL(
    dim_list=saved_args['dim_list'],
    device=device,
    use_not=saved_args['use_not'],
    use_skip=saved_args.get('use_skip', False), # use .get for safety
    estimated_grad=saved_args.get('estimated_grad', False),
    use_nlaf=saved_args.get('use_nlaf', False),
    alpha=saved_args.get('alpha', 0.999),
    beta=saved_args.get('beta', 8),
    gamma=saved_args.get('gamma', 1),
    distributed=False,
    is_rank0=True
)

# Load weights
# We need to handle potential 'module.' prefix if trained with DistributedDataParallel
new_state_dict = {}
for k, v in model_state_dict.items():
    name = k[7:] if k.startswith('module.') else k 
    new_state_dict[name] = v

rrl.net.load_state_dict(new_state_dict)
rrl.net.eval()
print("Model loaded successfully.")

# --- 3. Evaluate ---
print("Running evaluation...")
y_true_indices = []
y_pred_indices = []
y_pred_probs = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        
        # Forward pass
        # Output of rrl.net is logits (or something similar depending on the exact RRL version, usually logits)
        outputs = rrl.net(X_batch)
        
        # Apply softmax to get probabilities for the positive class (assuming index 1)
        # Check output shape. If 2 classes, outputs shape is (batch, 2)
        probs = torch.softmax(outputs, dim=1)
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        
        # True labels (y_batch is one-hot encoded in this setup)
        _, true_labels = torch.max(y_batch.to(device), 1)
        
        y_true_indices.extend(true_labels.cpu().numpy())
        y_pred_indices.extend(preds.cpu().numpy())
        y_pred_probs.extend(probs[:, 1].cpu().numpy()) # Probability of class 1

y_true = np.array(y_true_indices)
y_pred = np.array(y_pred_indices)
y_probs = np.array(y_pred_probs)

# Metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='binary') # Focus on positive class for banking dataset
cm = confusion_matrix(y_true, y_pred)

print("-" * 30)
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (Class 1): {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['No', 'Yes']))
print("-" * 30)

# --- 4. Visualization ---
print("Generating visualizations...")

# A. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted No', 'Predicted Yes'], 
            yticklabels=['Actual No', 'Actual Yes'])
plt.title('Confusion Matrix - RRL Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_path = os.path.join(PLOT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"Saved confusion matrix to {cm_path}")

# B. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
roc_path = os.path.join(PLOT_DIR, 'roc_curve.png')
plt.savefig(roc_path)
plt.close()
print(f"Saved ROC curve to {roc_path}")

# C. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
pr_path = os.path.join(PLOT_DIR, 'pr_curve.png')
plt.savefig(pr_path)
plt.close()
print(f"Saved Precision-Recall curve to {pr_path}")

print("Evaluation complete.")
