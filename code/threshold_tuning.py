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

import config
from dataset import CXRDataset
from model import CXRMultimodalModel


def collect_predictions(model, split_path, device):
    dataset = CXRDataset(split_path)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    labels, probabilities = [], []
    with torch.no_grad():
        for images, tabular, batch_labels, _ in loader:
            logits = model(images.to(device), tabular.to(device))
            labels.append(batch_labels.numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def select_threshold(truth, scores):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        score = f1_score(truth, scores >= threshold, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def calculate_metrics(truth, scores, threshold):
    predictions = scores >= threshold
    result = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
    }
    return result


def main():
    if not config.BEST_MODEL_PATH.is_file():
        raise SystemExit("ERROR: Best checkpoint not found. Run python train.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model = CXRMultimodalModel(checkpoint["tabular_dim"], len(config.LABEL_COLUMNS)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    validation_labels, validation_probabilities = collect_predictions(model, config.SPLIT_DIR / "val.csv", device)
    test_labels, test_probabilities = collect_predictions(model, config.SPLIT_DIR / "test.csv", device)
    thresholds = {}
    validation_metrics = {}
    test_metrics = {}
    for index, label in enumerate(config.LABEL_COLUMNS):
        threshold, _ = select_threshold(validation_labels[:, index], validation_probabilities[:, index])
        thresholds[label] = threshold
        validation_metrics[label] = calculate_metrics(validation_labels[:, index], validation_probabilities[:, index], threshold)
        test_metrics[label] = calculate_metrics(test_labels[:, index], test_probabilities[:, index], threshold)

    results = {
        "checkpoint": str(config.BEST_MODEL_PATH),
        "device": str(device),
        "threshold_selection": "validation_only",
        "thresholds": thresholds,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_macro_roc_auc": float(np.mean([item["roc_auc"] for item in test_metrics.values()])),
        "test_macro_average_precision": float(np.mean([item["average_precision"] for item in test_metrics.values()])),
        "test_macro_f1": float(np.mean([item["f1"] for item in test_metrics.values()])),
    }
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.METRICS_DIR / "threshold_tuning_metrics.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved threshold-tuning metrics to {output_path}")


if __name__ == "__main__":
    main()