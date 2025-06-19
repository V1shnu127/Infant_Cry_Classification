import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, hamming_loss, accuracy_score, roc_curve
)

# === SETTINGS ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
learning_rate = 1e-4
batch_size = 64
output_nodes = 2
max_coeffs = 20
max_len = 800

# === Dataset and DataLoader ===
train_data_path = "/kaggle/input/mfcc-bc2-202411071/mfcc/train"
test_data_path = "/kaggle/input/mfcc-bc2-202411071/mfcc/test"
validation_data_path = "/kaggle/input/mfcc-bc2-202411071/mfcc/val"

class PtDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.files = []
        for c in self.classes:
            c_dir = os.path.join(directory, c)
            c_files = [(os.path.join(c_dir, f), self.class_to_idx[c]) for f in os.listdir(c_dir)]
            self.files.extend(c_files)
        random.shuffle(self.files)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filepath, label = self.files[idx]
        try:
            mat_vals = scipy.io.loadmat(filepath)
            data = mat_vals['final']
            data = data.T  # ensure shape is (time, freq) = (800, 20)
    
            max_len = 800
            if max_len > data.shape[0]:
                pad_width = max_len - data.shape[0]
                data = np.pad(data, pad_width=((0, pad_width), (0, 0)), mode='constant')
            else:
                data = data[:max_len, :]
    
            data = torch.tensor(data, dtype=torch.float32)  # shape: (800, 20)
            data = data.unsqueeze(0)                        # shape: (1, 20, 800)
    
        except Exception as e:
            print(f"Error loading file {filepath}: {str(e)}")
            return None
    
        return data, label

train_dataset = PtDataset(train_data_path)
test_dataset = PtDataset(test_data_path)
val_dataset = PtDataset(validation_data_path)

class PtDataLoader(DataLoader):
    def __init__(self, directory, batch_size, shuffle=True):
        dataset = PtDataset(directory)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

train_dataloader = PtDataLoader(directory=train_data_path, batch_size=batch_size)
test_dataloader = PtDataLoader(directory=test_data_path, batch_size=batch_size)
val_dataloader = PtDataLoader(directory=validation_data_path, batch_size=batch_size)

train_count = len(train_dataset)
test_count = len(test_dataset)
val_count = len(val_dataset)
print(f"Train samples: {train_count}")
print(f"Test samples: {test_count}")
print(f"Validation samples: {val_count}")

# === CNN Model ===
class CNN(nn.Module):
    def __init__(self, num_classes=8, max_coeffs=20, max_len=800):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: (1, 20, 800), Output: (32, 20, 800)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),                 # Output: (32, 10, 400)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: (64, 10, 400)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),                 # Output: (64, 5, 200)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# Output: (128, 5, 200)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),                 # Output: (128, 2, 100)
        )

        # Calculate flattened size
        flattened_input = torch.zeros(1, 1, max_coeffs, max_len)
        flattened_output = self.features(flattened_input)
        flatten_size = flattened_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# === Model Initialization ===
model = CNN(num_classes=8, max_coeffs=max_coeffs, max_len=max_len).to(device)

# === Optimizer & Loss ===
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# === Training Loop ===
max_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0

    print(f"\nEpoch {epoch+1}/{num_epochs} — Training")
    for inputs, labels in tqdm(train_dataloader, desc="Training"):
        if inputs is None:
            continue

        inputs = inputs.to(device).float()  # (batch_size, 1, 20, 800)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels).item()

    train_acc = correct_train / train_count
    train_loss = running_loss / train_count

    # === Validation ===
    model.eval()
    correct_val = 0
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{num_epochs} — Validation")
        for inputs, labels in tqdm(val_dataloader, desc="Validating"):
            if inputs is None:
                continue

            inputs = inputs.to(device).float()
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels).item()

    val_acc = correct_val / val_count

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > max_acc:
        max_acc = val_acc
        torch.save(model, "best_model.pth")
        print("✅ Best model saved!")

print(f"\n🎯 Training complete. Best Validation Accuracy: {max_acc:.4f}")

# === Testing ===
model = torch.load("best_model.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()

pred_labels = []
act_labels = []
prob_scores = []

print("\n🔍 Testing model on test set")
with torch.no_grad():
    for inputs, labels in tqdm(test_dataloader, desc="Testing"):
        if inputs is None:
            continue
        inputs = inputs.to(device).float()  # (batch_size, 1, 20, 800)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        pred_labels.extend(preds.cpu().tolist())
        act_labels.extend(labels.cpu().tolist())
        prob_scores.extend(probs.cpu().tolist())

# === Convert to NumPy ===
pred_labels = np.array(pred_labels)
act_labels = np.array(act_labels)
prob_scores = np.array(prob_scores)

# === Confusion Matrix (8-class) ===
conf_mat = confusion_matrix(act_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, cmap="flare", annot=True, fmt="g",
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes,
            cbar_kws={"label": "Count"})
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (8-class)")
plt.savefig("ConfusionMatrix_8class.png")
plt.show()

# === Binary Confusion Matrix (Healthy vs Pathology) ===
healthy_idx = list(range(4))
pathology_idx = list(range(4, 8))

act_labels_bin = np.array([0 if i < 4 else 1 for i in act_labels])
pred_labels_bin = np.array([0 if i < 4 else 1 for i in pred_labels])
conf_mat_bin = confusion_matrix(act_labels_bin, pred_labels_bin)

plt.figure(figsize=(5, 4))
sns.heatmap(conf_mat_bin, cmap="flare", annot=True, fmt="g",
            xticklabels=["Healthy", "Pathology"],
            yticklabels=["Healthy", "Pathology"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Binary)")
plt.savefig("ConfusionMatrix_Binary.png")
plt.show()

# === METRICS ===
# 8-class metrics
print("\n📊 8-Class Metrics:")
print(f"Accuracy:           {accuracy_score(act_labels, pred_labels):.4f}")
print(f"F1 Score (macro):   {f1_score(act_labels, pred_labels, average='macro'):.4f}")
print(f"Precision (macro):  {precision_score(act_labels, pred_labels, average='macro', zero_division=0):.4f}")
print(f"Recall (macro):     {recall_score(act_labels, pred_labels, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss:       {hamming_loss(act_labels, pred_labels):.4f}")
print(f"MCC:                {matthews_corrcoef(act_labels, pred_labels):.4f}")

# Binary metrics
print("\n📊 Binary Metrics:")
print(f"Accuracy:           {accuracy_score(act_labels_bin, pred_labels_bin):.4f}")
print(f"F1 Score (macro):   {f1_score(act_labels_bin, pred_labels_bin, average='macro'):.4f}")
print(f"Precision (macro):  {precision_score(act_labels_bin, pred_labels_bin, average='macro', zero_division=0):.4f}")
print(f"Recall (macro):     {recall_score(act_labels_bin, pred_labels_bin, average='macro', zero_division=0):.4f}")
print(f"Hamming Loss:       {hamming_loss(act_labels_bin, pred_labels_bin):.4f}")
print(f"MCC:                {matthews_corrcoef(act_labels_bin, pred_labels_bin):.4f}")

# AUC & EER (only for binary)
prob_class1 = prob_scores[:, 1] if prob_scores.shape[1] > 1 else prob_scores
auc_score = roc_auc_score(act_labels_bin, prob_class1)
fpr, tpr, thresholds = roc_curve(act_labels_bin, prob_class1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

print(f"AUC:                {auc_score:.4f}")
print(f"EER:                {eer:.4f}")
