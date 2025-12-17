import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from rrl.models import RRL
from rrl.utils import DBEncoder
import os
import sys

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
except:
    pass
print(f"Using device: {device}")

# Load data
data_path = 'datasets/bank-marketing/bank-full.csv'
print(f"Loading data from {data_path}...")
try:
    df = pd.read_csv(data_path, sep=';')
except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
    sys.exit(1)

# Preprocess data
# Identify discrete and continuous columns
# RRL DBEncoder expects a feature info dataframe
# Column 0: feature name, Column 1: type ('discrete', 'continuous')

print("Preprocessing data...")
f_list = []
for col in df.columns[:-1]: # Exclude target 'y'
    if df[col].dtype == 'object':
        f_list.append([col, 'discrete'])
    else:
        f_list.append([col, 'continuous'])

f_df = pd.DataFrame(f_list)

# Initialize DBEncoder
db_enc = DBEncoder(f_df, discrete=False)
X_df = df.iloc[:, :-1]
y_df = df.iloc[:, -1:]

db_enc.fit(X_df, y_df)

X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoaders
batch_size = 64
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize RRL Model
discrete_flen = db_enc.discrete_flen
continuous_flen = db_enc.continuous_flen
output_dim = y.shape[1] # One-hot encoded length

# Structure from args.py default '5@64'
# We'll use a slightly larger structure given the dataset complexity
structure = [16, 16, 8] 
dim_list = [(discrete_flen, continuous_flen)] + structure + [output_dim]

print(f"RRL Structure: {dim_list}")

model_path = 'outputs/models/rrl_bank_model.pth'
log_path = 'outputs/models/rrl_bank_log.txt'

rrl = RRL(
    dim_list=dim_list,
    device=device,
    use_not=True,
    is_rank0=True,
    save_best=True,
    distributed=False,
    save_path=model_path,
    log_file=log_path
)

# Train
print("Starting training...")
rrl.train_model(
    data_loader=train_loader,
    valid_loader=test_loader,
    epoch=20,
    lr=0.01
)

# Test
print("Testing model...")
rrl.test(test_loader=test_loader, set_name='Test')

# Print Rules
print("Generating rules...")
rules_path = 'outputs/models/rrl_bank_rules.txt'
with open(rules_path, 'w') as f:
    rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=f, mean=db_enc.mean, std=db_enc.std)

print(f"Model saved to {model_path}")
print(f"Rules saved to {rules_path}")
print("Done.")
