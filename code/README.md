# Multimodal CXR Fusion

This project combines chest X-ray images with ICU tabular data to predict Atelectasis, Cardiomegaly, Edema, and Pneumonia. The image branch uses ImageNet-pretrained ResNet-50, the tabular branch uses an MLP, and learned gated fusion combines their 256-dimensional representations.

## Installation

From the `cxr_fusion` project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\code\requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your NVIDIA driver if the default PyTorch install is CPU-only. Training intentionally stops when CUDA is unavailable. Evaluation can run on CPU.

## Configuration

Open `config.py` and edit only these two paths:

```python
CSV_PATH = "C:/path/to/cxr_fusion/Final_Dataset.csv"
IMAGE_ROOT = "C:/path/to/cxr_fusion/Dataset"
```

Use forward slashes in Windows paths. `NUM_WORKERS=0` is intentional for Windows stability.

## Dataset layout

`Final_Dataset.csv` must be at the project root, and `Dataset/` must contain the complete image tree:

```text
cxr_fusion/
  Final_Dataset.csv
  Dataset/p10/p10002428/s123/image.jpg
  code/
```

The matching CSV value is `p10/p10002428/s123/image.jpg`, relative to `Dataset/`. Required columns are `subject_id`, `image_path`, the four labels, `heart_rate`, `spo2`, `sbp`, `dbp`, `gender`, and `age_at_imaging`.

## Commands

Run from `cxr_fusion/code` in this order:

```powershell
python data_prep.py
python sanity_checks.py --branch image
python sanity_checks.py --branch tabular
python train.py
python evaluate.py
```

## Clinical data statistics and bias screening

Run the descriptive audit from `cxr_fusion/code`:

```powershell
python clinical_data_audit.py
```

The script prints row and subject counts, vital-sign and age summaries, missingness, gender counts, label prevalence, and heuristic dataset-quality/bias signals. It also compares label prevalence across train/validation/test splits and writes the full report to `metrics/clinical_data_audit.json`.

Warnings are screening signals, not proof of discrimination or causal bias. Review group sizes, label quality, clinical context, and model performance by subgroup before drawing conclusions.

## Baseline models

The project also includes independent baselines for comparison with gated fusion. All use the same subject-disjoint splits, labels, preprocessing, class-weighted BCE loss, and early stopping.

From `cxr_fusion/code`:

```powershell
python train_baseline_resnet.py
python train_baseline_densenet.py
python train_baseline_efficientnetb3.py
python train_baseline_convnext_tiny.py
python train_baseline_mlp.py
python evaluate_baselines.py --model all
```

`train_baseline_resnet.py` trains an ImageNet-pretrained ResNet-50 using only images. `train_baseline_densenet.py` independently trains an ImageNet-pretrained DenseNet-121 using only images. `train_baseline_efficientnetb3.py` trains an ImageNet-pretrained EfficientNetB3 using only images. `train_baseline_convnext_tiny.py` trains an ImageNet-pretrained ConvNeXt Tiny using only images. All image baselines require CUDA. `train_baseline_mlp.py` trains an MLP using only the prepared clinical features and works on CPU or CUDA. Checkpoints are saved under `checkpoints/baselines/`, and comparable metrics are saved to `metrics/baseline_metrics.json`.

To compare all four image architectures:

```powershell
python train_baseline_resnet.py
python train_baseline_densenet.py
python train_baseline_efficientnetb3.py
python train_baseline_convnext_tiny.py
python evaluate_baselines.py --model image_models
```

To evaluate only one baseline:

```powershell
python evaluate_baselines.py --model image
python evaluate_baselines.py --model densenet
python evaluate_baselines.py --model efficientnetb3
python evaluate_baselines.py --model convnext_tiny
python evaluate_baselines.py --model clinical
```

`data_prep.py` validates every referenced image, converts labels as requested, clips vital signs, creates subject-disjoint 70/15/15 splits with seed 42, and fits imputation/standardization statistics on training rows only. It writes split CSVs and preprocessing metadata under `splits/`. The best checkpoint is `checkpoints/best_model.pt`; test metrics and probabilities are written under `metrics/`.

## Common errors

- **CSV file not found:** update `CSV_PATH` and confirm the file is named `Final_Dataset.csv`.
- **Image root not found or missing image files:** update `IMAGE_ROOT`; every `image_path` must resolve beneath that folder.
- **Missing required columns:** add the required CSV columns listed above.
- **Split files are missing:** run `python data_prep.py` first.
- **CUDA is unavailable during training:** install a CUDA-enabled PyTorch build, verify your NVIDIA driver and GPU, then rerun `python train.py`. CPU evaluation remains supported.
