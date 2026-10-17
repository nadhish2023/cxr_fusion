import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from main_model.code import config
from baselines.code.baseline_models import MLPBaseline, DenseNetBaseline, ResNetBaseline, EfficientNetB3Baseline, ConvNeXtTinyBaseline
from main_model.code.dataset import CXRDataset


BASELINE_DIR = config.BASELINE_CHECKPOINT_DIR
METRICS_DIR = config.BASELINE_METRICS_DIR
RESULT_STEMS = {
    "clinical": "mlp",
    "convnext_tiny": "convnext_tiny",
    "resnet": "resnet",
    "densenet": "densenet",
    "efficientnetb3": "efficientnetb3",
    "logistic_regression": "logistic_regression",
    "randomforest": "randomforest",
    "xgboost": "xgboost",
}


CLASSICAL_CLINICAL_MODELS = ["logistic_regression", "randomforest", "xgboost"]


def collect_classical_predictions(model_name, checkpoint_path):
    checkpoint = joblib.load(checkpoint_path)
    dataset = CXRDataset(config.SPLIT_DIR / "test.csv")
    rows = dataset.rows
    features = rows[checkpoint["feature_columns"]].to_numpy(dtype="float32")
    labels = rows[config.LABEL_COLUMNS].to_numpy(dtype="float32")
    models = checkpoint.get("models")
    if models is None:
        models = [checkpoint["model"]]

    if model_name == "logistic_regression":
        probabilities = checkpoint["model"].predict_proba(features)
    elif model_name == "randomforest":
        probabilities = np.column_stack(
            [model.predict_proba(features)[:, 1] for model in models]
        )
    else:
        probabilities = np.column_stack(
            [model.predict_proba(features)[:, 1] for model in models]
        )

    return labels, probabilities


def collect_predictions(model_name, checkpoint_path, device):
    if model_name in CLASSICAL_CLINICAL_MODELS:
        return collect_classical_predictions(model_name, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    dataset = CXRDataset(config.SPLIT_DIR / "test.csv")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    if model_name == "resnet":
        model = ResNetBaseline(len(config.LABEL_COLUMNS)).to(device)

    elif model_name == "densenet":
        model = DenseNetBaseline(len(config.LABEL_COLUMNS)).to(device)

    elif model_name == "efficientnetb3":
        model = EfficientNetB3Baseline(len(config.LABEL_COLUMNS)).to(device)

    elif model_name == "convnext_tiny":
        model = ConvNeXtTinyBaseline(len(config.LABEL_COLUMNS)).to(device)

    elif model_name == "clinical":
        model = MLPBaseline(
            checkpoint["tabular_dim"],
            len(config.LABEL_COLUMNS),
        ).to(device)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    labels = []
    probabilities = []

    with torch.no_grad():
        for images, tabular, targets, _ in loader:

            if model_name in ["resnet", "densenet", "efficientnetb3", "convnext_tiny"]:
                inputs = images.to(device)
            else:
                inputs = tabular.to(device)

            logits = model(inputs)

            labels.append(targets.numpy())
            probabilities.append(
                torch.sigmoid(logits).cpu().numpy()
            )

    return np.concatenate(labels), np.concatenate(probabilities)


def evaluate(labels, probabilities, device=None):
    predictions = (probabilities >= 0.5).astype(np.int32)

    results = {
        "threshold": 0.5,
        "device": str(device) if device is not None else None,
        "num_rows": int(len(labels)),
        "labels": {},
    }

    roc_auc_values = []
    average_precision_values = []
    f1_values = []
    valid_labels = []
    valid_predictions = []

    for index, label in enumerate(config.LABEL_COLUMNS):
        valid = np.isfinite(labels[:, index])
        truth = labels[valid, index].astype(np.int32)
        scores = probabilities[valid, index]
        predicted = predictions[valid, index]
        valid_labels.append(truth)
        valid_predictions.append(predicted)
        f1_values.append(f1_score(truth, predicted, zero_division=0))

        metrics = {
            "valid_count": int(valid.sum()),
            "positive_rate": float(truth.mean()),
            "accuracy": float(
                accuracy_score(truth, predicted)
            ),
            "precision": float(
                precision_score(
                    truth,
                    predicted,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    truth,
                    predicted,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    truth,
                    predicted,
                    zero_division=0,
                )
            ),
            "roc_auc": None,
            "average_precision": None,
        }

        # ROC-AUC and Average Precision require
        # both positive and negative samples.
        if len(np.unique(truth)) > 1:

            metrics["roc_auc"] = float(
                roc_auc_score(truth, scores)
            )

            metrics["average_precision"] = float(
                average_precision_score(truth, scores)
            )

            roc_auc_values.append(
                metrics["roc_auc"]
            )

            average_precision_values.append(
                metrics["average_precision"]
            )

        results["labels"][label] = metrics

    # Macro ROC-AUC
    results["macro_roc_auc"] = (
        float(np.mean(roc_auc_values))
        if roc_auc_values
        else None
    )

    # Macro Average Precision
    results["macro_average_precision"] = (
        float(np.mean(average_precision_values))
        if average_precision_values
        else None
    )

    # Macro F1
    flattened_labels = np.concatenate(valid_labels)
    flattened_predictions = np.concatenate(valid_predictions)
    results["macro_f1"] = float(
        np.mean(f1_values)
    )

    # Micro F1
    results["micro_f1"] = float(
        f1_score(
            flattened_labels,
            flattened_predictions,
            average="micro",
            zero_division=0,
        )
    )

    return results


def save_detailed_outputs(model_name, labels, probabilities, results):
    """Save model-specific metrics, predictions, and diagnostic plots."""
    result_stem = RESULT_STEMS[model_name]
    output_dir = METRICS_DIR / result_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = METRICS_DIR / f"{result_stem}_baseline_results.json"
    predictions_path = output_dir / f"{result_stem}_predictions.csv"

    prediction_frame = pd.DataFrame(
        labels,
        columns=[f"{label}_label" for label in config.LABEL_COLUMNS],
    )
    probability_frame = pd.DataFrame(
        probabilities,
        columns=[f"{label}_probability" for label in config.LABEL_COLUMNS],
    )
    pd.concat([prediction_frame, probability_frame], axis=1).to_csv(
        predictions_path,
        index=False,
    )

    roc_figure, roc_axes = plt.subplots(2, 2, figsize=(12, 9))
    pr_figure, pr_axes = plt.subplots(2, 2, figsize=(12, 9))
    confusion_figure, confusion_axes = plt.subplots(2, 2, figsize=(10, 8))
    for index, label in enumerate(config.LABEL_COLUMNS):
        valid = np.isfinite(labels[:, index])
        truth = labels[valid, index].astype(np.int32)
        scores = probabilities[valid, index]
        predicted = (scores >= 0.5).astype(np.int32)
        roc_axis = roc_axes.flat[index]
        pr_axis = pr_axes.flat[index]
        confusion_axis = confusion_axes.flat[index]

        if len(np.unique(truth)) > 1:
            false_positive_rate, true_positive_rate, _ = roc_curve(truth, scores)
            precision, recall, _ = precision_recall_curve(truth, scores)
            roc_axis.plot(false_positive_rate, true_positive_rate, label=f"AUC={roc_auc_score(truth, scores):.3f}")
            pr_axis.plot(recall, precision, label=f"AP={average_precision_score(truth, scores):.3f}")
            roc_axis.legend(loc="lower right")
            pr_axis.legend(loc="lower left")
        else:
            roc_axis.text(0.5, 0.5, "Insufficient class variation", ha="center", va="center")
            pr_axis.text(0.5, 0.5, "Insufficient class variation", ha="center", va="center")

        roc_axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        roc_axis.set_title(label)
        roc_axis.set_xlabel("False positive rate")
        roc_axis.set_ylabel("True positive rate")
        roc_axis.grid(alpha=0.25)
        pr_axis.set_title(label)
        pr_axis.set_xlabel("Recall")
        pr_axis.set_ylabel("Precision")
        pr_axis.grid(alpha=0.25)
        confusion_axis.imshow(
            confusion_matrix(truth, predicted, labels=[0, 1]),
            cmap="Blues",
        )
        confusion_axis.set_title(label)
        confusion_axis.set_xlabel("Predicted label")
        confusion_axis.set_ylabel("True label")
        confusion_axis.set_xticks([0, 1])
        confusion_axis.set_yticks([0, 1])

        matrix = confusion_matrix(truth, predicted, labels=[0, 1])
        for row in range(2):
            for column in range(2):
                confusion_axis.text(column, row, matrix[row, column], ha="center", va="center")

    for figure, filename, title in [
        (roc_figure, "roc_curves.png", f"{result_stem} ROC curves"),
        (pr_figure, "precision_recall_curves.png", f"{result_stem} precision-recall curves"),
        (confusion_figure, "confusion_matrices.png", f"{result_stem} confusion matrices"),
    ]:
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved detailed results to {report_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved plots to {output_dir}")
    return report_path


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ResNet-50, DenseNet-121, "
            "and clinical baselines."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "resnet",
            "densenet",
            "efficientnetb3",
            "convnext_tiny",
            "logistic_regression",
            "randomforest",
            "xgboost",
            "clinical",
            "all",
            "image_models",
            "clinical_models",
        ],
        default="all",
        help=(
            "Which baseline to evaluate. "
            "'image_models' evaluates image models; "
            "'clinical_models' evaluates clinical baselines."
        ),
    )

    args = parser.parse_args()

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Determine which models to evaluate
    if args.model == "all":
        models = [
            "resnet",
            "densenet",
            "efficientnetb3",
            "convnext_tiny",
            "clinical",
            "logistic_regression",
            "randomforest",
            "xgboost",
        ]

    elif args.model == "image_models":
        models = [
            "resnet",
            "densenet",
            "efficientnetb3",
            "convnext_tiny",
        ]

    elif args.model == "clinical_models":
        models = [
            "clinical",
            "logistic_regression",
            "randomforest",
            "xgboost",
        ]

    else:
        models = [args.model]

    # ---------------------------------------------------------
    # IMPORTANT:
    # Model names used internally do not necessarily match
    # checkpoint filenames.
    #
    # Model name "resnet" loads "baseline_resnet.pt",
    # "densenet" loads "baseline_densenet.pt",
    # "clinical" loads "baseline_mlp.pt".
    # ---------------------------------------------------------

    checkpoint_names = {
        "resnet": "baseline_resnet.pt",
        "densenet": "baseline_densenet.pt",
        "efficientnetb3": "baseline_efficientnetb3.pt",
        "convnext_tiny": "baseline_convnext_tiny.pt",
        "clinical": "baseline_mlp.pt",
        "logistic_regression": "baseline_logistic_regression.joblib",
        "randomforest": "baseline_randomforest.joblib",
        "xgboost": "baseline_xgboost.joblib",
    }

    model_descriptions = {
        "resnet": "ResNet-50",
        "densenet": "DenseNet-121",
        "efficientnetb3": "EfficientNetB3",
        "convnext_tiny": "ConvNeXt Tiny",
        "clinical": "MLP (Clinical)",
        "logistic_regression": "Logistic Regression (Clinical)",
        "randomforest": "Random Forest (Clinical)",
        "xgboost": "XGBoost (Clinical)",
    }

    # ---------------------------------------------------------
    # Evaluate each selected model
    # ---------------------------------------------------------

    for model_name in models:

        checkpoint_path = (
            BASELINE_DIR
            / checkpoint_names[model_name]
        )

        print()
        print("=" * 70)
        print(f"Evaluating: {model_descriptions[model_name]}")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 70)

        # Check checkpoint exists
        if not checkpoint_path.is_file():

            raise SystemExit(
                f"\nERROR: Missing {model_name} checkpoint:\n"
                f"{checkpoint_path}\n\n"
                f"Expected checkpoint filename:\n"
                f"{checkpoint_names[model_name]}\n\n"
                f"Train that baseline first."
            )

        # Collect predictions
        labels, probabilities = collect_predictions(
            model_name,
            checkpoint_path,
            device,
        )

        # Calculate metrics
        result = evaluate(
            labels,
            probabilities,
            device,
        )
        result["model"] = model_name
        result["checkpoint"] = str(
            checkpoint_path
        )
        save_detailed_outputs(model_name, labels, probabilities, result)

        # Print summary
        print()
        print(f"{model_descriptions[model_name]} baseline results:")
        print(
            f"  Macro ROC-AUC: "
            f"{result['macro_roc_auc']:.4f}"
        )
        print(
            f"  Macro Average Precision: "
            f"{result['macro_average_precision']:.4f}"
        )
        print(
            f"  Macro F1: "
            f"{result['macro_f1']:.4f}"
        )
        print(
            f"  Micro F1: "
            f"{result['micro_f1']:.4f}"
        )

if __name__ == "__main__":
    main()
