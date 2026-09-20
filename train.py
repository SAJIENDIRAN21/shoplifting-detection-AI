"""Train and validate the BiLSTM pose classifier."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.bilstm_classifier import BiLSTMPoseClassifier


PROJECT_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total_samples = 0
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            probabilities = model(features).squeeze(1)
            total_loss += loss_fn(probabilities, targets).item() * targets.size(0)
            total_correct += ((probabilities >= 0.5) == targets.bool()).sum().item()
            total_samples += targets.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BiLSTM shoplifting-pose classifier.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8, help="Early-stop after this many non-improving validation epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    sequence_dir = PROJECT_DIR / "data" / "sequences"
    x_path, y_path = sequence_dir / "X.npy", sequence_dir / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Dataset files are absent. Run: python extract_dataset.py")
    x_data = np.load(x_path).astype(np.float32)
    y_data = np.load(y_path).astype(np.float32).reshape(-1)
    if x_data.ndim != 3 or x_data.shape[1:] != (30, 51) or len(x_data) != len(y_data):
        raise ValueError("Expected X.npy shape (N, 30, 51) and a matching y.npy")
    if len(y_data) < 10 or len(np.unique(y_data)) != 2:
        raise ValueError("Training requires at least 10 sequences spanning normal (0) and concealment (1) classes")

    indices = np.arange(len(y_data))
    train_indices, validation_indices = train_test_split(indices, test_size=0.20, stratify=y_data, random_state=args.seed)
    train_dataset = TensorDataset(torch.from_numpy(x_data[train_indices]), torch.from_numpy(y_data[train_indices]))
    validation_dataset = TensorDataset(torch.from_numpy(x_data[validation_indices]), torch.from_numpy(y_data[validation_indices]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPoseClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCELoss()
    best_path = PROJECT_DIR / "models" / "best_bilstm.pth"
    best_loss, stagnant_epochs = float("inf"), 0
    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_correct = total_samples = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(features).squeeze(1)
            loss = loss_fn(probabilities, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += ((probabilities >= 0.5) == targets.bool()).sum().item()
            total_samples += targets.size(0)
        train_loss, train_accuracy = total_loss / total_samples, total_correct / total_samples
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_fn, device)
        for name, value in zip(history, (train_loss, validation_loss, train_accuracy, validation_accuracy)):
            history[name].append(value)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f}, acc {train_accuracy:.2%} | val loss {validation_loss:.4f}, acc {validation_accuracy:.2%}")

        if validation_loss < best_loss:
            best_loss, stagnant_epochs = validation_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation_loss": validation_loss}, best_path)
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best validation loss: {best_loss:.4f}")
                break

    result_dir = PROJECT_DIR / "results"
    result_dir.mkdir(exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCELoss")
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(result_dir / "training_curves.png", dpi=160)
    print(f"Best weights: {best_path}; curves: {result_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
