import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

from dataset import make_dataloaders
from lstm import LSTM

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

CSV        = "../data/results/gold_base_stationary_dropna.csv"
SPLIT_JSON = "../data/results/split_definition.json"
TARGET_COL = "Gold Futures (COMEX) | log_return"
SEQ_LEN    = 20
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 5e-4
HIDDEN     = 64
NUM_LAYERS = 2
DROPOUT    = 0.2
LOSS_SCALE = 1e6  # Multiply losses to make them visible as integers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(model, optimizer, loss_fn, loader, training):
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

def plot_losses(train_loss_list, val_loss_list, best_epoch):
    epochs = range(1, len(train_loss_list) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_list, color='steelblue', label='Train Loss')
    plt.plot(epochs, val_loss_list, color='tomato', label='Val Loss')
    plt.axvline(x=best_epoch, color='green', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('losses.png', dpi=150)

def run(csv_file, split_json, seq_len, batch_size, target_col, epochs, lr, hidden, num_layers, dropout, loss_scale):
    train_loader, val_loader, test_loader, n_features = make_dataloaders(
        csv_file, split_json, seq_len=seq_len, batch_size=batch_size, target_col=target_col
    )

    model = LSTM(
        input_size=n_features,
        hidden_size=hidden,
        output_size=1,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 0
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, optimizer, loss_fn, train_loader, training=True)
        val_loss   = run_epoch(model, optimizer, loss_fn, val_loader, training=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch

        gap = (val_loss - train_loss) * loss_scale
        marker = " <<" if gap > 0 else ""
        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss*loss_scale:7.2f}  val={val_loss*loss_scale:7.2f}  gap={gap:+7.2f}{marker}")
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss*loss_scale:.2f} (×{str(1/loss_scale)})")
    plot_losses(train_loss_list, val_loss_list, best_epoch)

    # Evaluate on test split using the best checkpoint
    model.load_state_dict(best_state)
    test_loss = run_epoch(model, optimizer, loss_fn, test_loader, training=False)
    print(f"\nTest MSE: {test_loss*loss_scale:.2f} (×{str(1/loss_scale)})")

def main():
    run(CSV, SPLIT_JSON, SEQ_LEN, BATCH_SIZE, TARGET_COL, EPOCHS, LR, HIDDEN, NUM_LAYERS, DROPOUT, LOSS_SCALE)

if __name__ == "__main__":
    main()