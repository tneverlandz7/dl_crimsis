import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("CRISISMMD_ROOT", ROOT / "data" / "CrisisMMD_v2.0"))
SPLIT_DIRNAME = "crisismmd_datasplit_agreed_label"

FEATURES_DIR = ROOT / "features"
IMAGE_FEATURES = FEATURES_DIR / "resnet18_imagenet1k_v1.pt"
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"
ENV_TRACKER = ROOT / "env_tracker.csv"

TASKS = ("informative", "humanitarian")
SPLITS = ("train", "dev", "test")
LABELS = {
    "informative": ["informative", "not_informative"],
    "humanitarian": [
        "affected_individuals",
        "infrastructure_and_utility_damage",
        "rescue_volunteering_or_donation_effort",
        "other_relevant_information",
        "not_humanitarian",
    ],
}

# Fixed by the coursework brief
MAX_TOKENS = 64
EMBED_DIM = 128
IMAGE_FEATURE_DIM = 512
BATCH_SIZE = 64
MAX_EPOCHS = 20
PATIENCE = 3
SEEDS = (0, 1, 2)

# Carbon intensity for C = E x I. Default: UK DESNZ 2024 grid factor (annual average).
CARBON_INTENSITY_G_PER_KWH = float(os.environ.get("CARBON_INTENSITY_G_PER_KWH", 207.05))
CODECARBON_COUNTRY = os.environ.get("CODECARBON_COUNTRY", "GBR")

# Fallback engineering estimate (guidance s5.2): E = t * (Pcpu*ucpu + Pgpu*ugpu + Pram) * PUE / 1000
CPU_POWER_W = float(os.environ.get("CPU_POWER_W", 45))
CPU_UTIL = float(os.environ.get("CPU_UTIL", 0.5))
GPU_POWER_W = float(os.environ.get("GPU_POWER_W", 0))
GPU_UTIL = float(os.environ.get("GPU_UTIL", 0.5))
RAM_POWER_W_PER_GB = 0.375
PUE = float(os.environ.get("PUE", 1.0))
