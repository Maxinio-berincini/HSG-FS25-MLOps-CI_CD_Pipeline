import mlflow
import os

import torch
from mlflow.tracking import MlflowClient
from torch import nn
import numpy as np

# import data iterator and eval method
from data_utils import test_iterator

# load config
from config import REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, PRODUCTION_ALIAS

# load test condition specification
from test_specification import TestCondition
FILE_PATH = "test_condition.txt"
with open(FILE_PATH, 'r') as f:
    CONDITION_SPECIFICATION = f.readline().strip()


# load data_utils
from data_utils import estimate_required_samples, calculate_achievable_confidence

# set up device & loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.CrossEntropyLoss()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()


# method to evaluate model with confidence intervals following ease.ml
def evaluate_model_with_confidence_interval(model_uri, confidence=0.9999):
    # load the model
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
                # first element is logits
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    sample_accuracies = [1 if pred == label else 0 for pred, label in zip(all_preds, all_labels)]

    # using bootstrap method to calculate confidence interval
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
        # half-width of the confidence interval == actual observed error
        "eval_epsilon": (upper_bound - lower_bound) / 2,
        "confidence": confidence
    }


# get challenger alias
challenger_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias=CHALLENGER_ALIAS)
challenger_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHALLENGER_ALIAS}"


#Setup: set target epsilon and confidence level when calculating confidence intervals
TARGET_EPSILON = 0.02  # 2% error tolerance
TARGET_CONFIDENCE = 0.90  # 99.99% confidence level


# first, check if test set is large enough or if confidence has to be adjusted with given epsilon
test_set_size = len(list(test_iterator.dataset))
required_size = estimate_required_samples(TARGET_EPSILON, 1 - TARGET_CONFIDENCE)

if test_set_size >= required_size:
    EPSILON = TARGET_EPSILON
    CONFIDENCE = TARGET_CONFIDENCE
    print( f"Test set size {test_set_size} is sufficient for epsilon={EPSILON} and confidence={CONFIDENCE}.")
else:
    achievable_confidence = calculate_achievable_confidence(test_set_size, TARGET_EPSILON)
    bare_minimum_confidence = 0.85

    if achievable_confidence < bare_minimum_confidence:  # Set a minimum acceptable confidence of bare_minimum_confidence% for calculating confidence intervalls %
        print(f"Test set size {test_set_size} provides only {achievable_confidence:.4f} confidence " 
              f"with epsilon={TARGET_EPSILON}. This is below minimum acceptable level ({bare_minimum_confidence:.4f}).")
        print(f"Required size for target confidence: {required_size}")
        exit(1)

    else:
        EPSILON = TARGET_EPSILON
        CONFIDENCE = achievable_confidence
        print(f"Warning: Using reduced confidence {CONFIDENCE:.4f} instead of target {TARGET_CONFIDENCE}")
        print(f"To achieve target confidence, increase test set from {test_set_size} to {required_size} samples")



# secondly, evaluate challenger model with statistical confidence interval

challenger_metrics = evaluate_model_with_confidence_interval(challenger_uri, confidence=CONFIDENCE)
print(f"Challenger model accuracy: {challenger_metrics['mean_accuracy']:.3f} "
      f"± {challenger_metrics['eval_epsilon']:.3f} (confidence: {challenger_metrics['confidence']})")

if challenger_metrics['eval_epsilon'] > EPSILON:
    print(
        f"Warning: Achieved precision ({challenger_metrics['eval_epsilon']:.4f}) is worse than target ({EPSILON:.4f})")

# then, evaluate production model with statistical confidence interval
try:
    prod_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS)
    prod_uri = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
    production_metrics = evaluate_model_with_confidence_interval(prod_uri, confidence=CONFIDENCE)
except Exception:
    production_metrics = {'accuracy': -1.0, 'lower_bound': -1.0, 'upper_bound': -1.0}
    print("No production model found.")

# Evaluate the specified test condition
condition = TestCondition(CONDITION_SPECIFICATION)
is_valid, message = condition.evaluate(challenger_metrics, production_metrics)


if is_valid:
    print(f"Challenger model meets the condition: {message}")
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

    # add description to model version
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=challenger_mv.version,
        description=f"Model accuracy: {challenger_metrics['mean_accuracy']:.3f} "
                    f"± {challenger_metrics['eval_epsilon']:.3f} (confidence: {challenger_metrics['confidence']})"
    )

    print(f"Promoted version {challenger_mv.version} to production (score {challenger_metrics['mean_accuracy']:.3f})")
    exit(0)
else:
    print(
        f"Challenger ({challenger_metrics['mean_accuracy']:.3f}) ≤ production ({production_metrics['mean_accuracy']:.3f}), not promoted")
    exit(1)
