import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

try:
    from . import config
    from .dataset import CXRDataset
    from .model import CXRMultimodalModel
except ImportError:
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
    train_valid = torch.isfinite(train_labels)
    positive_counts = torch.nan_to_num(train_labels, nan=0.0).sum(dim=0)
    negative_counts = (train_valid & (train_labels == 0)).sum(dim=0).float()
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
    best_val_roc_auc = -float("inf")
    print(f"Training positive counts: {positive_counts.detach().cpu().tolist()}")
    print(f"Using BCE pos_weight: {[round(value, 3) for value in pos_weight.detach().cpu().tolist()]}")
    print(f"Image backbone frozen until epoch {config.UNFREEZE_IMAGE_EPOCH}")

    for epoch in range(1, config.EPOCHS + 1):
        if not image_backbone_unfrozen and epoch >= config.UNFREEZE_IMAGE_EPOCH:
            final_stage_parameters = list(model.image_encoder.features[5:].parameters())
            final_stage_parameter_ids = {id(parameter) for parameter in final_stage_parameters}
            for parameter in final_stage_parameters:
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in final_stage_parameter_ids], "lr": config.LEARNING_RATE},
                {"params": final_stage_parameters, "lr": config.IMAGE_BACKBONE_LEARNING_RATE},
            ], weight_decay=config.WEIGHT_DECAY)
            image_backbone_unfrozen = True
            print(f"Unfroze ConvNeXt Tiny final stages at epoch {epoch}; backbone_lr={config.IMAGE_BACKBONE_LEARNING_RATE}")
        model.train()
        train_loss = 0.0
        for images, tabular, labels, _ in train_loader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, tabular)
            valid = torch.isfinite(labels)
            loss_values = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.nan_to_num(labels, nan=0.0), pos_weight=pos_weight, reduction="none"
            )
            loss = loss_values.masked_select(valid).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        model.eval()
        val_loss = 0.0
        validation_labels, validation_probabilities = [], []
        with torch.no_grad():
            for images, tabular, labels, _ in val_loader:
                images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
                logits = model(images, tabular)
                valid = torch.isfinite(labels)
                loss_values = nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.nan_to_num(labels, nan=0.0), pos_weight=pos_weight, reduction="none"
                )
                val_loss += loss_values.masked_select(valid).sum().item()
                validation_labels.append(labels.cpu().numpy())
                validation_probabilities.append(torch.sigmoid(logits).cpu().numpy())
        train_loss /= len(train_set)
        validation_labels = np.concatenate(validation_labels)
        validation_probabilities = np.concatenate(validation_probabilities)
        val_loss /= np.isfinite(validation_labels).sum()
        valid_roc_auc_scores = []
        for index in range(len(config.LABEL_COLUMNS)):
            valid_mask = np.isfinite(validation_labels[:, index])
            if valid_mask.sum() <= 1:
                continue
            y_true = validation_labels[valid_mask, index]
            y_score = validation_probabilities[valid_mask, index]
            if len(np.unique(y_true)) > 1:
                valid_roc_auc_scores.append(roc_auc_score(y_true, y_score))
        validation_roc_auc = float(np.mean(valid_roc_auc_scores)) if valid_roc_auc_scores else float("nan")
        print(f"Epoch {epoch:02d}/{config.EPOCHS}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_roc_auc={validation_roc_auc:.4f}")
        if validation_roc_auc > best_val_roc_auc:
            best_val_roc_auc = validation_roc_auc
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "backbone": "convnext_tiny", "tabular_dim": len(config.TABULAR_FEATURE_COLUMNS), "labels": config.LABEL_COLUMNS, "epoch": epoch, "val_loss": val_loss, "val_macro_roc_auc": validation_roc_auc}, config.BEST_MODEL_PATH)
            print(f"Saved best model to {config.BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}; best_val_macro_roc_auc={best_val_roc_auc:.4f}")
                break


if __name__ == "__main__":
    main()
