import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from dataset import CXRDataset
from model import CXRMultimodalModel


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is unavailable. Training requires an NVIDIA GPU with a working CUDA PyTorch install.")
    device = torch.device("cuda")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    train_set = CXRDataset(config.SPLIT_DIR / "train.csv", train=True)
    val_set = CXRDataset(config.SPLIT_DIR / "val.csv")
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    model = CXRMultimodalModel(len(config.TABULAR_FEATURE_COLUMNS), len(config.LABEL_COLUMNS)).to(device)
    train_labels = torch.tensor(train_set.rows[config.LABEL_COLUMNS].to_numpy(dtype="float32"), device=device)
    positive_counts = train_labels.sum(dim=0)
    negative_counts = train_labels.shape[0] - positive_counts
    pos_weight = (negative_counts / positive_counts.clamp_min(1.0)).clamp(max=10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for parameter in model.image_encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    image_backbone_unfrozen = False
    print(f"Training positive counts: {positive_counts.detach().cpu().tolist()}")
    print(f"Using BCE pos_weight: {[round(value, 3) for value in pos_weight.detach().cpu().tolist()]}")
    print(f"Image backbone frozen until epoch {config.UNFREEZE_IMAGE_EPOCH}")

    for epoch in range(1, config.EPOCHS + 1):
        if not image_backbone_unfrozen and epoch >= config.UNFREEZE_IMAGE_EPOCH:
            layer4_parameters = list(model.image_encoder.layer4.parameters())
            layer4_parameter_ids = {id(parameter) for parameter in layer4_parameters}
            for parameter in layer4_parameters:
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in layer4_parameter_ids], "lr": config.LEARNING_RATE},
                {"params": layer4_parameters, "lr": config.IMAGE_BACKBONE_LEARNING_RATE},
            ], weight_decay=config.WEIGHT_DECAY)
            image_backbone_unfrozen = True
            print(f"Unfroze ResNet layer4 at epoch {epoch}; backbone_lr={config.IMAGE_BACKBONE_LEARNING_RATE}")
        model.train()
        train_loss = 0.0
        for images, tabular, labels, _ in train_loader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images, tabular), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, tabular, labels, _ in val_loader:
                images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
                logits = model(images, tabular)
                val_loss += criterion(logits, labels).item() * images.size(0)
        train_loss /= len(train_set)
        val_loss /= len(val_set)
        print(f"Epoch {epoch:02d}/{config.EPOCHS}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "tabular_dim": len(config.TABULAR_FEATURE_COLUMNS), "labels": config.LABEL_COLUMNS, "epoch": epoch, "val_loss": val_loss}, config.BEST_MODEL_PATH)
            print(f"Saved best model to {config.BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}; best_val_loss={best_val_loss:.4f}")
                break


if __name__ == "__main__":
    main()
