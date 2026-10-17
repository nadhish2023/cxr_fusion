import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from main_model.code import config
from baselines.code.baseline_models import DenseNetBaseline
from main_model.code.dataset import CXRDataset


BASELINE_DIR = config.BASELINE_CHECKPOINT_DIR
CHECKPOINT_PATH = BASELINE_DIR / "baseline_densenet.pt"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is unavailable. DenseNet-121 baseline training requires CUDA.")
    device = torch.device("cuda")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    train_set = CXRDataset(config.SPLIT_DIR / "train.csv", train=True)
    val_set = CXRDataset(config.SPLIT_DIR / "val.csv")
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    model = DenseNetBaseline(len(config.LABEL_COLUMNS)).to(device)
    labels = torch.tensor(train_set.rows[config.LABEL_COLUMNS].to_numpy(dtype="float32"), device=device)
    valid_labels = torch.isfinite(labels)
    positive_counts = torch.nan_to_num(labels, nan=0.0).sum(dim=0)
    negative_counts = (valid_labels & (labels == 0)).sum(dim=0).float()
    pos_weight = (negative_counts / positive_counts.clamp_min(1.0)).clamp(max=10.0)
    validation_label_count = int(np.isfinite(val_set.rows[config.LABEL_COLUMNS].to_numpy(dtype="float32")).sum())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    best_val_loss = float("inf")
    patience = 0
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"DenseNet-121 baseline positive counts: {positive_counts.detach().cpu().tolist()}")
    print(f"DenseNet-121 baseline pos_weight: {[round(value, 3) for value in pos_weight.detach().cpu().tolist()]}")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, _, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            valid = torch.isfinite(targets)
            loss_values = nn.functional.binary_cross_entropy_with_logits(
                model(images),
                torch.nan_to_num(targets, nan=0.0),
                pos_weight=pos_weight,
                reduction="none",
            )
            loss = loss_values.masked_select(valid).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                valid = torch.isfinite(targets)
                loss_values = nn.functional.binary_cross_entropy_with_logits(
                    model(images),
                    torch.nan_to_num(targets, nan=0.0),
                    pos_weight=pos_weight,
                    reduction="none",
                )
                val_loss += loss_values.masked_select(valid).sum().item()
        train_loss /= len(train_set)
        val_loss /= validation_label_count
        print(f"Epoch {epoch:02d}/{config.EPOCHS}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save({"model_state": model.state_dict(), "labels": config.LABEL_COLUMNS, "epoch": epoch, "val_loss": val_loss}, CHECKPOINT_PATH)
            print(f"Saved baseline_densenet to {CHECKPOINT_PATH}")
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}; best_val_loss={best_val_loss:.4f}")
                break


if __name__ == "__main__":
    main()