import argparse
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
from baseline_models import MLPBaseline, DenseNetBaseline, ResNetBaseline, EfficientNetB3Baseline, ConvNeXtTinyBaseline
from dataset import CXRDataset


BASELINE_DIR = config.CHECKPOINT_DIR / "baselines"


def collect_predictions(model_name, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    dataset = CXRDataset(config.SPLIT_DIR / "test.csv")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    if model_name == "image":
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

            if model_name in ["image", "densenet", "efficientnetb3", "convnext_tiny"]:
                inputs = images.to(device)
            else:
                inputs = tabular.to(device)

            logits = model(inputs)

            labels.append(targets.numpy())
            probabilities.append(
                torch.sigmoid(logits).cpu().numpy()
            )

    return np.concatenate(labels), np.concatenate(probabilities)


def evaluate(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(np.int32)

    results = {
        "threshold": 0.5,
        "labels": {},
    }

    roc_auc_values = []
    average_precision_values = []

    for index, label in enumerate(config.LABEL_COLUMNS):

        truth = labels[:, index].astype(np.int32)
        scores = probabilities[:, index]
        predicted = predictions[:, index]

        metrics = {
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
    results["macro_f1"] = float(
        f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    # Micro F1
    results["micro_f1"] = float(
        f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        )
    )

    return results


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
            "image",
            "densenet",
            "efficientnetb3",
            "convnext_tiny",
            "clinical",
            "all",
            "image_models",
        ],
        default="all",
        help=(
            "Which baseline to evaluate. "
            "'image_models' evaluates all image-based models."
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
            "image",
            "densenet",
            "efficientnetb3",
            "convnext_tiny",
            "clinical",
        ]

    elif args.model == "image_models":
        models = [
            "image",
            "densenet",
            "efficientnetb3",
            "convnext_tiny",
        ]

    else:
        models = [args.model]

    # ---------------------------------------------------------
    # IMPORTANT:
    # Model names used internally do not necessarily match
    # checkpoint filenames.
    #
    # Model name "image" loads "baseline_resnet.pt",
    # "densenet" loads "baseline_densenet.pt",
    # "clinical" loads "baseline_mlp.pt".
    # ---------------------------------------------------------

    checkpoint_names = {
        "image": "baseline_resnet.pt",
        "densenet": "baseline_densenet.pt",
        "efficientnetb3": "baseline_efficientnetb3.pt",
        "convnext_tiny": "baseline_convnext_tiny.pt",
        "clinical": "baseline_mlp.pt",
    }

    model_descriptions = {
        "image": "ResNet-50",
        "densenet": "DenseNet-121",
        "efficientnetb3": "EfficientNetB3",
        "convnext_tiny": "ConvNeXt Tiny",
        "clinical": "MLP (Clinical)",
    }

    all_results = {
        "device": str(device),
        "num_rows": len(
            CXRDataset(
                config.SPLIT_DIR / "test.csv"
            )
        ),
        "models": {},
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
        )

        # Store checkpoint path
        result["checkpoint"] = str(
            checkpoint_path
        )

        # Store results
        all_results["models"][model_name] = result

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

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    config.METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        config.METRICS_DIR
        / "baseline_metrics.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            all_results,
            handle,
            indent=2,
        )

    # ---------------------------------------------------------
    # Print complete JSON
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLETE RESULTS")
    print("=" * 70)

    print(
        json.dumps(
            all_results,
            indent=2,
        )
    )

    print()
    print(
        f"Saved baseline metrics to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
