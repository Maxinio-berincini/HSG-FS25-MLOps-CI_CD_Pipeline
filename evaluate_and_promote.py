import mlflow
import os

import torch
from mlflow.tracking import MlflowClient
from torch import nn

import random
import numpy as np

# import data iterator and eval method
from data_utils import test_iterator

# load config
from config import REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, PRODUCTION_ALIAS

## load data_utils
from data_utils import is_test_set_sufficiently_large

# set up device & loss
DEVICE = torch.device("cuda")
CRITERION = nn.CrossEntropyLoss()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()

## method to evaluate model with confidence intervals following ease.ml
def evaluate_model_with_confidence_interval(model_uri, confidence=0.9999):
    #load the model
    model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_iterator:
            inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0] #first element is logits

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    sample_accuracies = [1 if pred == label else 0 for pred, label in zip(all_preds, all_labels)]

    ## using bootstrap method to calculate confidence interval 
    n_bootstrap = 10000
    bootstrap_accuracies = []
    n_samples = len(sample_accuracies)

    for _ in range(n_bootstrap):
        sample_items = np.random.choice(range(n_samples), n_samples, replace=True)
        bootstrap_sample = [sample_accuracies[i] for i in sample_items]
        bootstrap_accuracies.append(np.mean(bootstrap_sample))

    # Calculate the confidence interval
    alpha = 1 - confidence
    lower_bound = np.percentile(bootstrap_accuracies, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_accuracies, 100 * (1 - alpha / 2))
    mean_accuracy = np.mean(sample_accuracies)

    return {
        "mean_accuracy": mean_accuracy,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "eval_epsilon": (upper_bound - lower_bound) / 2, # half-width of the confidence interval == actual observed error 
        "confidence": confidence
    }

# method to compare statistics to be significant
def is_model_significantly_better(challenger_metrics, production_metrics, min_acc_improvement=0.01, allow_non_sig_improvements = False):
    
    effective_min_improvent = max(min_acc_improvement, 2 * challenger_metrics['eval_epsilon'])


    # Check if the confidence intervals overlap
    if challenger_metrics['lower_bound'] > production_metrics['upper_bound']:
        return True, "Challenger is significantly better"

    if challenger_metrics['mean_accuracy'] - production_metrics['mean_accuracy'] > effective_min_improvent:
        return True, "Challenger is better however not significant" if allow_non_sig_improvements else False , "Challenger is not significantly better"
    
    return False, "Challenger is not significantly better than production"


# get challenger alias
challenger_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias=CHALLENGER_ALIAS)
challenger_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHALLENGER_ALIAS}"


## testing with confidence intervals and statistical significance

EPSILON = 0.02 # 2% error tolerance
CONFIDENCE = 0.99999 # 99.99% confidence level
DELTA = 1-CONFIDENCE # 99.99% confidence level

# first, check if test set is large enough
is_sufficient, test_set_size, required_size = is_test_set_sufficiently_large(EPSILON, DELTA, test_iterator= test_iterator)
if not is_sufficient:
    print(f"Test set size {test_set_size} is not sufficient for epsilon={EPSILON}, delta={DELTA}. "
          f"Required size: {required_size}.")
    print(f"Please increase the test set size or decrease the epsilon and delta values.")
    exit(1)

# secondly, evaluate challenger model with statistical significance
challenger_metrics = evaluate_model_with_confidence_interval(challenger_uri, confidence=CONFIDENCE)
print(f"Challenger model accuracy: {challenger_metrics['mean_accuracy']:.3f} "
      f"± {challenger_metrics['eval_epsilon']:.3f} (confidence: {challenger_metrics['confidence']})")

if challenger_metrics['eval_epsilon'] > EPSILON:
    print(f"Warning: Achieved precision ({challenger_metrics['eval_epsilon']:.4f}) is worse than target ({EPSILON:.4f})")

# then, evaluate production model with statistical significance if exists
try:
    prod_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS)
    prod_uri = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
    production_metrics = evaluate_model_with_confidence_interval(prod_uri, confidence=CONFIDENCE)
except Exception:
    production_metrics = {'accuracy': -1.0, 'lower_bound': -1.0, 'upper_bound': -1.0}
    print("No production model found.")

## compare models with statistical guarantees
is_better, message = is_model_significantly_better(challenger_metrics, production_metrics, allow_non_sig_improvements=False)

if is_better:

    # clear old "production" alias
    try:
        client.delete_registered_model_alias(REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS)
    except Exception:
        pass

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=challenger_mv.version
    )
    print(f"Promoted version {challenger_mv.version} to production (score {challenger_metrics['mean_accuracy']:.3f})")
    exit(0)
else:
    print(f"Challenger ({challenger_metrics['mean_accuracy']:.3f}) ≤ production ({production_metrics['mean_accuracy']:.3f}), not promoted")
    exit(1)