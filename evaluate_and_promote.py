import mlflow
import os

import torch
from mlflow.tracking import MlflowClient
from torch import nn

# import data iterator and eval method
from data_utils import test_iterator
from train import evaluate

# load config
from config import REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, PRODUCTION_ALIAS

# set up device & loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.CrossEntropyLoss()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()


# download model from MLflow and evaluate
def evaluate_model(model_uri):
    # load the model
    model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    # evaluate with test iterator (CIFAR-10 test set)
    loss, acc = evaluate(model, test_iterator, CRITERION, DEVICE)
    return acc


# get challenger alias
chall_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias=CHALLENGER_ALIAS)
chall_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHALLENGER_ALIAS}"

# clear old "production" alias
try:
    client.delete_registered_model_alias(REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS)
except Exception:
    pass

# evaluate challenger
challenger_score = evaluate_model(chall_uri)

# evaluate production model if exists
try:
    prod_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS)
    prod_uri = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
    prod_score = evaluate_model(prod_uri)
except Exception:
    prod_score = -1.0

# swap aliases if challenger is better
if challenger_score > prod_score:
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=chall_mv.version
    )
    print(f"Promoted version {chall_mv.version} to production (score {challenger_score:.3f})")
    exit(0)
else:
    print(f"Challenger ({challenger_score:.3f}) ≤ production ({prod_score:.3f}), not promoted")
    exit(1)
