import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

import config
from dataset import CXRDataset
from model import CXRMultimodalModel


def main():
    if not config.BEST_MODEL_PATH.is_file():
        raise SystemExit("ERROR: Best checkpoint not found. Run python train.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CXRDataset(config.SPLIT_DIR / "test.csv")
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
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
    results = {"device": str(device), "num_rows": int(len(dataset)), "labels": {}}
    for index, label in enumerate(config.LABEL_COLUMNS):
        truth, score = labels[:, index], probabilities[:, index]
        metrics = {"positive_rate": float(truth.mean())}
        if len(np.unique(truth)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(truth, score))
            metrics["average_precision"] = float(average_precision_score(truth, score))
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
        results["labels"][label] = metrics
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.METRICS_DIR / "test_metrics.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    pd.DataFrame(probabilities, columns=[f"{label}_probability" for label in config.LABEL_COLUMNS]).to_csv(config.METRICS_DIR / "test_predictions.csv", index=False)
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
