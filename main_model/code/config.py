from pathlib import Path

# Edit only these two paths for a new machine.
CSV_PATH = Path("C:/Users/Nadhish SN/Desktop/Project-1/cxr_fusion/Final_Dataset.csv")
IMAGE_ROOT = Path("C:/Users/Nadhish SN/Desktop/Project-1/cxr_fusion/Dataset")

SEED = 42
NUM_WORKERS = 0
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
UNFREEZE_IMAGE_EPOCH = 3
LEARNING_RATE = 1e-4
IMAGE_BACKBONE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
TUNE_THRESHOLDS_ON_VALIDATION = True
LABEL_POLICY = "masked_definite"

LABEL_COLUMNS = ["Atelectasis", "Cardiomegaly", "Edema", "Pneumonia"]
VITAL_COLUMNS = ["heart_rate", "spo2", "sbp", "dbp"]
TABULAR_NUMERIC_COLUMNS = VITAL_COLUMNS + ["age_at_imaging"]
TABULAR_FEATURE_COLUMNS = [
    "heart_rate", "spo2", "sbp", "dbp", "age_at_imaging",
    "heart_rate_missing", "spo2_missing", "sbp_missing", "dbp_missing",
    "gender_M", "gender_F", "gender_unknown",
]
CLIP_RANGES = {
    "heart_rate": (30.0, 220.0),
    "spo2": (50.0, 100.0),
    "sbp": (60.0, 260.0),
    "dbp": (30.0, 150.0),
}

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(CSV_PATH).expanduser().resolve().parent
MAIN_MODEL_ROOT = CODE_DIR.parent
OUTPUT_DIR = MAIN_MODEL_ROOT / "outputs"
SPLIT_DIR = OUTPUT_DIR / "splits"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
BASELINE_OUTPUT_DIR = PROJECT_ROOT / "baselines" / "outputs"
BASELINE_CHECKPOINT_DIR = BASELINE_OUTPUT_DIR / "checkpoints"
BASELINE_METRICS_DIR = BASELINE_OUTPUT_DIR / "metrics"
PREPROCESSING_PATH = SPLIT_DIR / "preprocessing.json"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model_convnext_fusion.pt"
