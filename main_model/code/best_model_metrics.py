import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

try:
    from . import config
    from .dataset import CXRDataset
    from .model import CXRMultimodalModel
except ImportError:
    import config
    from dataset import CXRDataset
    from model import CXRMultimodalModel


def main():
    if not config.BEST_MODEL_PATH.is_file():
        raise SystemExit("ERROR: Best checkpoint not found. Run python train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CXRDataset(config.SPLIT_DIR / "test.csv")
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=False)
    model = CXRMultimodalModel(checkpoint["tabular_dim"], len(config.LABEL_COLUMNS)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_labels, all_probabilities = [], []
    with torch.no_grad():
        for images, tabular, labels, _ in loader:
            logits = model(images.to(device), tabular.to(device))
            all_labels.append(labels.numpy())
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())

    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    predictions = (probabilities >= 0.5).astype(np.int32)
    results = {
        "checkpoint": str(config.BEST_MODEL_PATH),
        "device": str(device),
        "num_rows": int(len(dataset)),
        "threshold": 0.5,
        "labels": {},
    }
    roc_auc_values, average_precision_values = [], []
    for index, label in enumerate(config.LABEL_COLUMNS):
        valid = np.isfinite(labels[:, index])
        truth = labels[valid, index].astype(np.int32)
        scores = probabilities[valid, index]
        predicted = predictions[:, index]
        metrics = {
            "eligible_rows": int(valid.sum()),
            "positive_rate": float(truth.mean()) if len(truth) else None,
            "accuracy": float(accuracy_score(truth, predicted)),
            "precision": float(precision_score(truth, predicted, zero_division=0)),
            "recall": float(recall_score(truth, predicted, zero_division=0)),
            "f1": float(f1_score(truth, predicted, zero_division=0)),
            "roc_auc": None,
            "average_precision": None,
        }
        if len(np.unique(truth)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(truth, scores))
            metrics["average_precision"] = float(average_precision_score(truth, scores))
            roc_auc_values.append(metrics["roc_auc"])
            average_precision_values.append(metrics["average_precision"])
        results["labels"][label] = metrics

    results["macro_roc_auc"] = float(np.mean(roc_auc_values)) if roc_auc_values else None
    results["macro_average_precision"] = float(np.mean(average_precision_values)) if average_precision_values else None
    results["macro_f1"] = float(np.mean([
        f1_score(labels[np.isfinite(labels[:, index]), index], predictions[np.isfinite(labels[:, index]), index], zero_division=0)
        for index in range(len(config.LABEL_COLUMNS))
    ]))
    results["micro_f1"] = float(f1_score(
        np.concatenate([labels[np.isfinite(labels[:, index]), index] for index in range(len(config.LABEL_COLUMNS))]),
        np.concatenate([predictions[np.isfinite(labels[:, index]), index] for index in range(len(config.LABEL_COLUMNS))]),
        average="binary", zero_division=0,
    ))

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.METRICS_DIR / "best_model_metrics.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved best-model metrics to {output_path}")


if __name__ == "__main__":
    main()