import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    from . import config
except ImportError:
    import config


def load_predictions(path):
    frame = pd.read_csv(path)
    probability_columns = [f"{label}_probability" for label in config.LABEL_COLUMNS]
    missing_probabilities = [column for column in probability_columns if column not in frame]
    if missing_probabilities:
        raise ValueError("Missing columns: " + ", ".join(missing_probabilities))

    label_columns = [f"{label}_label" for label in config.LABEL_COLUMNS]
    if all(column in frame for column in label_columns):
        labels = frame[label_columns].to_numpy(dtype=float)
    else:
        split_frame = pd.read_csv(config.SPLIT_DIR / "test.csv")
        if len(split_frame) != len(frame) or not all(label in split_frame for label in config.LABEL_COLUMNS):
            raise ValueError(
                "Prediction file has no labels, and test.csv cannot provide matching labels. "
                "Run evaluate.py again to regenerate test_predictions.csv."
            )
        labels = split_frame[config.LABEL_COLUMNS].to_numpy(dtype=float)

    probabilities = frame[
        probability_columns
    ].to_numpy(dtype=float)
    return labels, probabilities


def plot_roc(labels, probabilities, output_path):
    figure, axis = plt.subplots(figsize=(8, 6))
    for index, label in enumerate(config.LABEL_COLUMNS):
        valid = np.isfinite(labels[:, index])
        truth = labels[valid, index]
        scores = probabilities[valid, index]
        if len(np.unique(truth)) < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(truth, scores)
        score = roc_auc_score(truth, scores)
        axis.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"{label} (AUC={score:.3f})")
    axis.plot([0, 1], [0, 1], color="0.65", linestyle=":", label="Chance")
    axis.set(title="Test-set ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_precision_recall(labels, probabilities, output_path):
    figure, axis = plt.subplots(figsize=(8, 6))
    for index, label in enumerate(config.LABEL_COLUMNS):
        valid = np.isfinite(labels[:, index])
        truth = labels[valid, index]
        scores = probabilities[valid, index]
        if len(np.unique(truth)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(truth, scores)
        score = average_precision_score(truth, scores)
        axis.plot(recall, precision, linewidth=2, label=f"{label} (AP={score:.3f})")
    axis.set(title="Test-set precision-recall curves", xlabel="Recall", ylabel="Precision")
    axis.legend(loc="best", fontsize=9)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_confusion_matrices(labels, probabilities, threshold, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    for index, (axis, label) in enumerate(zip(axes.flat, config.LABEL_COLUMNS)):
        valid = np.isfinite(labels[:, index])
        matrix = confusion_matrix(labels[valid, index], probabilities[valid, index] >= threshold)
        ConfusionMatrixDisplay(matrix, display_labels=["Negative", "Positive"]).plot(
            ax=axis, cmap="Blues", colorbar=False, values_format="d"
        )
        axis.set_title(f"{label} (threshold={threshold:.2f})")
    figure.suptitle("Test-set confusion matrices", y=1.01)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_metric_summary(labels, probabilities, output_path):
    values = []
    for index in range(len(config.LABEL_COLUMNS)):
        valid = np.isfinite(labels[:, index])
        truth = labels[valid, index]
        scores = probabilities[valid, index]
        if len(np.unique(truth)) > 1:
            values.append([
                roc_auc_score(truth, scores),
                average_precision_score(truth, scores),
            ])
        else:
            values.append([np.nan, np.nan])

    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(config.LABEL_COLUMNS))
    width = 0.36
    axis.bar(positions - width / 2, [row[0] for row in values], width, label="ROC-AUC")
    axis.bar(positions + width / 2, [row[1] for row in values], width, label="Average precision")
    axis.set(
        title="Test-set ranking metrics",
        ylabel="Score",
        xticks=positions,
        xticklabels=config.LABEL_COLUMNS,
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Create evaluation plots from test_predictions.csv.")
    parser.add_argument("--predictions", type=Path, default=config.METRICS_DIR / "test_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=config.METRICS_DIR / "plots")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    labels, probabilities = load_predictions(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_roc(labels, probabilities, args.output_dir / "roc_curves.png")
    plot_precision_recall(labels, probabilities, args.output_dir / "precision_recall_curves.png")
    plot_confusion_matrices(labels, probabilities, args.threshold, args.output_dir / "confusion_matrices.png")
    plot_metric_summary(labels, probabilities, args.output_dir / "metric_summary.png")
    print(f"Saved evaluation plots to {args.output_dir}")


if __name__ == "__main__":
    main()