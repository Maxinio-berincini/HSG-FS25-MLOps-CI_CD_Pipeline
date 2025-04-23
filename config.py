import os

from dotenv import load_dotenv

# load .env
load_dotenv()

# set global experiment config

# experiment and model names
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "alexnet_auto_deploy_test-4")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "alexnet_model-4")
RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "challenger")

# model registry aliases
CHALLENGER_ALIAS = os.getenv("MLFLOW_CHALLENGER_ALIAS", "challenger")
PRODUCTION_ALIAS = os.getenv("MLFLOW_PRODUCTION_ALIAS", "production")

# environment spec and directories
CONDA_ENV_FILE = os.getenv("CONDA_ENV_FILE", "environment.yml")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "model_artifacts")
