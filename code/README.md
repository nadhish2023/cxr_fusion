# CXR Fusion Baseline Progress

This project predicts four chest X-ray findings from MIMIC-CXR images and ICU clinical data:

`Atelectasis`, `Cardiomegaly`, `Edema`, and `Pneumonia`.

The baseline suite establishes image-only and clinical-only reference points before comparing them with the multimodal gated-fusion model.

## Current Results

The table below is generated from the latest `metrics/baseline_metrics.json` report.

Evaluation set: **2,862 test rows**

Decision threshold: **0.5**

Device: **CUDA**
Split policy: **subject-disjoint train/validation/test splits**

| Rank | Baseline | Input | Macro ROC-AUC | Macro AP | Macro F1 | Micro F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | ConvNeXt Tiny | Image | **0.7096** | **0.5203** | **0.5546** | **0.5663** |
| 2 | ResNet-50 | Image | 0.6927 | 0.4950 | 0.5418 | 0.5464 |
| 3 | DenseNet-121 | Image | 0.6865 | 0.4936 | 0.5366 | 0.5419 |
| 4 | EfficientNetB3 | Image | 0.6908 | 0.4890 | 0.5435 | 0.5474 |
| 5 | Logistic Regression | Clinical | 0.5397 | 0.3594 | 0.4285 | 0.4347 |
| 6 | MLP | Clinical | 0.5378 | 0.3584 | **0.4422** | **0.4451** |
| 7 | XGBoost | Clinical | 0.5288 | 0.3523 | 0.4072 | 0.4133 |
| 8 | Random Forest | Clinical | 0.5204 | 0.3428 | 0.3410 | 0.3522 |

**Metric key:** AP = Average Precision. Higher is better for every metric shown.

## Pathology-Level ROC-AUC

| Baseline | Atelectasis | Cardiomegaly | Edema | Pneumonia |
|---|---:|---:|---:|---:|
| ConvNeXt Tiny | **0.7136** | **0.6748** | **0.7610** | **0.6889** |
| ResNet-50 | 0.6986 | 0.6456 | 0.7516 | 0.6750 |
| DenseNet-121 | 0.6788 | 0.6547 | 0.7449 | 0.6676 |
| EfficientNetB3 | 0.6965 | 0.6614 | 0.7325 | 0.6728 |
| Logistic Regression | 0.5055 | 0.5416 | 0.5729 | 0.5387 |
| MLP | 0.4960 | 0.5420 | 0.5709 | 0.5424 |
| XGBoost | 0.4885 | 0.5359 | 0.5574 | 0.5334 |
| Random Forest | 0.4880 | 0.5239 | 0.5363 | 0.5332 |

## Progress Summary

- **Best image baseline:** ConvNeXt Tiny, with macro ROC-AUC `0.7096`.
- **Best clinical ranking model:** Logistic Regression, with macro ROC-AUC `0.5397`.
- **Best clinical thresholded classification:** MLP, with macro F1 `0.4422` and micro F1 `0.4451`.
- **Most visually difficult finding:** Pneumonia has the lowest image-baseline ROC-AUC range.
- **Strongest image finding:** Edema is highest across the image models, reaching `0.7610` with ConvNeXt Tiny.
- The image-only models substantially outperform the clinical-only models on ranking metrics. Clinical features remain important comparison points for testing whether gated fusion adds complementary information.

These results are baseline references, not final claims. Model selection and conclusions should use a held-out test set, confidence intervals or repeated splits where possible, and subgroup analysis from the clinical audit.

## Reproduce The Results

Run from `cxr_fusion/code`.

### Prepare data

```powershell
python data_prep.py
python sanity_checks.py --branch image
python sanity_checks.py --branch tabular
```

### Train image baselines

```powershell
python train_baseline_resnet.py
python train_baseline_densenet.py
python train_baseline_efficientnetb3.py
python train_baseline_convnext_tiny.py
```

Image baselines require CUDA and use ImageNet-pretrained backbones.

### Train clinical baselines

```powershell
python train_baseline_mlp.py
python train_baseline_tree.py --model all
```

The classical trainer produces Logistic Regression, Random Forest, and XGBoost checkpoints.

### Evaluate

```powershell
python evaluate_baselines.py --model image_models
python evaluate_baselines.py --model clinical_models
python evaluate_baselines.py --model all
```

The complete report is written to `metrics/baseline_metrics.json`. Checkpoints are stored under `checkpoints/baselines/`.

## Model Inventory

| Model | Type | Checkpoint |
|---|---|---|
| ResNet-50 | Image neural baseline | `checkpoints/baselines/baseline_resnet.pt` |
| DenseNet-121 | Image neural baseline | `checkpoints/baselines/baseline_densenet.pt` |
| EfficientNetB3 | Image neural baseline | `checkpoints/baselines/baseline_efficientnetb3.pt` |
| ConvNeXt Tiny | Image neural baseline | `checkpoints/baselines/baseline_convnext_tiny.pt` |
| MLP | Clinical neural baseline | `checkpoints/baselines/baseline_mlp.pt` |
| Logistic Regression | Clinical linear baseline | `checkpoints/baselines/baseline_logistic_regression.joblib` |
| Random Forest | Clinical tree baseline | `checkpoints/baselines/baseline_random_forest.joblib` |
| XGBoost | Clinical boosting baseline | `checkpoints/baselines/baseline_xgboost.joblib` |

## Project Configuration

Edit `config.py` when using a new machine:

```python
CSV_PATH = "C:/path/to/cxr_fusion/Final_Dataset.csv"
IMAGE_ROOT = "C:/path/to/cxr_fusion/Dataset"
```

The dataset uses ImageNet normalization, prepared clinical features, clipped vital signs, and training-only preprocessing statistics. `NUM_WORKERS=0` is intentional for Windows stability.

## Clinical Audit

Run:

```powershell
python clinical_data_audit.py
```

The audit writes `metrics/clinical_data_audit.json` and reports row counts, subject counts, missingness, demographic summaries, label prevalence, and split-level quality signals. Warnings are screening signals, not proof of discrimination or causality.

## Dependencies

Install from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\code\requirements.txt
```

Use a CUDA-enabled PyTorch build appropriate for the installed NVIDIA driver for image and multimodal training. Evaluation can run on CPU, although the current metrics report was generated on CUDA.
