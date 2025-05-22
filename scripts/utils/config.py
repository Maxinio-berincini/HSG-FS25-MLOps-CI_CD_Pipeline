import os
from pathlib import Path
from dotenv import load_dotenv

# determine project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# load .env
load_dotenv(PROJECT_ROOT / "configs" / ".env")

# set global experiment config

# experiment and model names
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "alexnet_auto_deploy_demo")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "alexnet_model-demo")
RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "challenger")

# model registry aliases
CHALLENGER_ALIAS = os.getenv("MLFLOW_CHALLENGER_ALIAS", "challenger")
PRODUCTION_ALIAS = os.getenv("MLFLOW_PRODUCTION_ALIAS", "production")

# environment spec and directories
CONDA_ENV_FILE = os.getenv("CONDA_ENV_FILE", str(PROJECT_ROOT / "configs" / "environment.yml"))
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", PROJECT_ROOT / "model_artifacts")

# Hugging Face repo
HF_REPO_ID = os.getenv("HF_REPO_ID", "Maxinio-Berincini/HSG-FS25-MLOps-CI_CD_Pipeline")

# Training Parameters
LEARNING_RATE = os.getenv("LEARNING_RATE", 0.001)
EPOCHS = os.getenv("NUM_EPOCHS", 8)

# Data Parameters
OUTPUT_DIM = os.getenv("OUTPUT_DIM", 10)  # number of classes in the dataset
BATCH_SIZE = os.getenv("BATCH_SIZE", 256)  # batch size for training
VALID_RATIO = os.getenv("VALID_RATIO", 0.9)  # ratio of training data to validation data
SEED = os.getenv("SEED", 1234)  # seed for reproducibility
DATA_ROOT = os.getenv("DATA_ROOT", "../../data")  # root directory for the dataset

# Evaluation Parameters
TARGET_EPSILON = os.getenv("TARGET_EPSILON", 0.015)  # target error tolerance in evaluation
TARGET_CONFIDENCE = os.getenv("TARGET_CONFIDENCE", 0.99)  # target confidence level ==> directly influences the confidence interval width
