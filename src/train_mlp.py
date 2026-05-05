import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json

from pathlib import Path
from optuna.importance import get_param_importances
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, recall_score

data = pd.read_csv("data/heart_2020_cleaned.csv")
y = LabelEncoder().fit_transform(data['HeartDisease'])
data = data.drop(columns=['HeartDisease'])

age_map = {val: i for i, val in enumerate(sorted(data['AgeCategory'].unique()))}
data['AgeCategory'] = data['AgeCategory'].map(age_map)

gen_health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
data['GenHealth'] = data['GenHealth'].map(gen_health_map)

numeric_col = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory', 'GenHealth']
categorical_col = [c for c in data.columns if c not in numeric_col]

for col in numeric_col:
    upper = data[col].quantile(0.99)
    data.loc[data[col] > upper, col] = upper

X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=42, stratify=y)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_train = encoder.fit_transform(X_train[categorical_col])
X_cat_val = encoder.transform(X_val[categorical_col])

scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_train[numeric_col])
X_num_val = scaler.transform(X_val[numeric_col])

X_train = np.hstack([X_cat_train, X_num_train])
X_val = np.hstack([X_cat_val, X_num_val])

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TabularDataset(X_train, y_train)
val_ds = TabularDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Linear(64, num_classes)
        )
    
    def forward(self, X):
        return self.net(X)

device = torch.device("cuda")

weights = torch.tensor([1.0, 10.0]).to(device)

def eval_epoch(model, loader, criterion):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)

            loss = criterion(outputs, y)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    metrics = {
        'acc': acc,
        'f1': f1,
        'roc_auc': roc_auc
        }

    return total_loss / len(loader), metrics

def train_model(model, train_loader, val_loader, input_dim, epochs=100, patience=5):
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []
    history = {'acc': [], 'f1': [], 'roc_auc': []}

    best_val_loss = float('inf')
    epoch_no_improve = 0

    os.makedirs("ml", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss, metrics = eval_epoch(model, val_loader, criterion)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        for key in metrics:
            history[key].append(metrics[key])

        print(
            f'Epoch {epoch + 1:02d} | '
            f'Train Loss: {avg_train_loss:.4f} | '
            f'Val Loss: {avg_val_loss:.4f} | '
            f'Acc: {metrics["acc"]:.4f} | '
            f'F1: {metrics["f1"]:.4f} | '
            f'AUC: {metrics["roc_auc"]:.4f}'
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epoch_no_improve = 0

            torch.save(model.state_dict(), "ml/model_mlp.pth")

            meta = {
                "input_dim": input_dim,
                "num_classes": 2
            }

            with open("ml/mlp_meta.json", "w") as f:
                json.dump(meta, f)

        else:
            epoch_no_improve += 1

            if epoch_no_improve >= patience:
                print(f'\nEarly stopping triggered at epoch {epoch + 1}')
                model.load_state_dict(torch.load("ml/model_mlp.pth", weights_only=True))
                break

    return model, train_losses, val_losses, history
torch.manual_seed(42)


print(f'\n=== Training MLP (Device: {device}) ===')

in_features = X_train.shape[1]
model = MLP(in_dim=in_features, num_classes=2).to(device)

model, train_losses, val_losses, history = train_model(
    model,
    train_loader,
    val_loader,
    input_dim=X_train.shape[1],
    epochs=100,
    patience=7
)

plt.figure(figsize=(18, 5))
epochs_range = range(len(train_losses))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, label='Train Loss', lw=2)
plt.plot(epochs_range, val_losses, label='Validation Loss', lw=2)
plt.title('Loss Curve', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(epochs_range, history['acc'], label='Val Accuracy', color='green', lw=2)
plt.plot(epochs_range, history['roc_auc'], label='Val ROC-AUC', color='blue', lw=2)
plt.title('Acc & AUC Metrics', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(epochs_range, history['f1'], label='Val F1 Score', color='red', lw=2)
plt.title('F1 Score Curve', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

best_epoch = np.argmin(val_losses)
print("\n" + "="*30)
print("MLP BEST RESULTS:")
print(f"Best Epoch: {best_epoch + 1}")
print(f"ROC-AUC:    {history['roc_auc'][best_epoch]:.4f}")
print(f"Accuracy:   {history['acc'][best_epoch]:.4f}")
print(f"F1 Score:   {history['f1'][best_epoch]:.4f}")
print("="*30)


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in val_loader:
        X = X.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Heart Disease', 'Heart Disease'])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Confusion Matrix: MLP')
plt.show()

recall = recall_score(all_labels, all_preds)
print(f"Recall (Чувствительность): {recall:.4f}")
print("Это процент больных, которых модель смогла найти.")