
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
CSV_FILE = HERE / "breast_cancer.csv"
MODEL_FILE = HERE / "breast_cancer_mlp.pth"


class CsvDataset(Dataset):
    """Read numeric feature columns and a final 0/1 label from a CSV file."""

    def __init__(self, csv_path):
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row.
            for row in reader:
                rows.append([float(value) for value in row])

        data = torch.tensor(rows, dtype=torch.float32)
        self.features = data[:, :-1]
        self.labels = data[:, -1].unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class MLP(nn.Module):
    """A small fully connected neural network for binary classification."""

    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),  # Output a logit; BCEWithLogitsLoss applies sigmoid safely.
        )

    def forward(self, x):
        return self.layers(x)


def write_scaled_csv(features, labels, train_indices, val_indices):
    """Fit scaling on training rows only, then save scaled rows for CsvDataset."""
    scaler = StandardScaler()
    scaler.fit(features[train_indices])
    scaled = scaler.transform(features).astype(np.float32)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([f"feature_{i}" for i in range(scaled.shape[1])] + ["label"])
        for x, label in zip(scaled, labels):
            writer.writerow([*x.tolist(), int(label)])
    return scaled


def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)
    return correct / total


def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    data = load_breast_cancer()
    features = data.data
    labels = data.target
    all_indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        all_indices, test_size=0.2, random_state=7, stratify=labels
    )

    scaled_features = write_scaled_csv(features, labels, train_indices, val_indices)
    dataset = CsvDataset(CSV_FILE)
    train_set = torch.utils.data.Subset(dataset, train_indices.tolist())
    val_set = torch.utils.data.Subset(dataset, val_indices.tolist())
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = MLP(input_size=dataset.features.shape[1])
    print("Params:", sum(parameter.numel() for parameter in model.parameters()))
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_accuracy = 0.0
    best_state = None
    best_epoch = 0
    patience = 7
    epochs_without_improvement = 0
    training_losses = []
    validation_accuracies = []

    for epoch in range(40):
        model.train()
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()  # Clear gradients from the previous batch.
            logits = model(batch_features)  # Forward pass.
            loss = loss_function(logits, batch_labels)  # Measure prediction error.
            loss.backward()  # Backpropagation computes gradients.
            optimizer.step()  # Adam updates the weights.
            total_loss += loss.item() * len(batch_labels)

        average_loss = total_loss / len(train_set)
        val_accuracy = accuracy(model, val_loader)
        training_losses.append(average_loss)
        validation_accuracies.append(val_accuracy)
        print(f"Epoch {epoch + 1:02d}: train loss={average_loss:.4f}, val accuracy={val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping: validation accuracy did not improve for {patience} epochs.")
            break

    # Keep the best validation checkpoint, not simply the last epoch.
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_FILE)
    best_accuracy = accuracy(model, val_loader)
    print(f"Best val accuracy {best_accuracy:.4f} at epoch {best_epoch}")
    print("Training losses recorded:", len(training_losses))
    print("Validation accuracies recorded:", len(validation_accuracies))

    # Loading with weights_only is supported by current PyTorch versions.
    reloaded = MLP(input_size=dataset.features.shape[1])
    reloaded.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
    reloaded.eval()
    reloaded_accuracy = accuracy(reloaded, val_loader)
    print(f"Reloaded model accuracy: {reloaded_accuracy:.4f} (identical: {reloaded_accuracy == best_accuracy})")
    print("Saved model:", MODEL_FILE)
    print("Saved CSV dataset:", CSV_FILE)

    # Use the exact same split and train-only scaling for a fair baseline.
    baseline = LogisticRegression(max_iter=1000, random_state=7)
    baseline.fit(scaled_features[train_indices], labels[train_indices])
    baseline_predictions = baseline.predict(scaled_features[val_indices])
    print(f"Logistic regression val accuracy: {accuracy_score(labels[val_indices], baseline_predictions):.4f}")


if __name__ == "__main__":
    main()
