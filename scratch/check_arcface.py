import os
import json
import torch
import numpy as np
import yaml
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from function.models import UnifiedModel
from src.train_model import MultiClassManifestDataset

with open("../setting.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PT_DIR = cfg["path"]["pt"]
OUTPUT_DIR = cfg["path"]["output"]
TYPE = cfg["model"]["type"]
BSZ = 4 # Small batch size for printing

# Load manifest
manifest_file = os.path.join(OUTPUT_DIR, f"identification_{TYPE}_folds.json")
with open(manifest_file, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

fold_data = manifest["fold_1"]
train_ds = MultiClassManifestDataset(fold_data["train"]["paths"][:16], fold_data["train"]["labels"][:16], desc="Test Load")
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BSZ, shuffle=False)

sample_x, _ = train_ds[0]
input_dim = sample_x.shape[-1]
num_classes = len(manifest["class_mapping"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnifiedModel(
    input_dim=input_dim,
    hidden_dim=64,
    num_layers=2,
    num_classes=num_classes,
    arch="gru",
    use_arcface=True
).to(device)

model.train()
for xb, yb in train_loader:
    xb = xb.to(device)
    yb = yb.to(device)
    
    # Forward with ArcFace margin
    logits = model(xb, yb)
    
    preds = torch.argmax(logits, dim=1)
    
    print("\n--- Batch ---")
    print("Labels (yb):", yb.cpu().numpy())
    print("Predictions:", preds.cpu().numpy())
    print("Logits:")
    print(logits.detach().cpu().numpy())
    
    # Check cosine values before scaling
    cosine = torch.matmul(torch.nn.functional.normalize(model.enc(xb)), torch.nn.functional.normalize(model.head.weight).t())
    print("Cosines (before scale & margin):")
    print(cosine.detach().cpu().numpy())
    break
