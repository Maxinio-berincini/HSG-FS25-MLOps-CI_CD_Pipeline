import logging
import os
import time

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.notebook import tqdm, trange

# load config
from config import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    RUN_NAME,
    CHALLENGER_ALIAS,
    CONDA_ENV_FILE,
    ARTIFACT_DIR,
)
# import data iterator and model
from data_utils import train_iterator, valid_iterator, test_iterator
from model import AlexNet, count_parameters, initialize_parameters

# configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(EXPERIMENT_NAME)

# ensure the model_artifacts directory exists
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# set up logging
logging.getLogger("mlflow").setLevel(logging.INFO)


# decays learning rate
class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch
        r = curr_iter / self.num_iter
        return [
            base_lr * (self.end_lr / base_lr) ** r
            for base_lr in self.base_lrs
        ]


# wrap dataloader to allow continuous get_batch() calls
class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels = next(self._iterator)
        return inputs, labels

    def get_batch(self):
        return next(self)


# compute classification accuracy
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# train model for one epoch and return loss and accuracy
def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for x, y in tqdm(iterator, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred, _ = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# evaluate model and return average loss and accuracy
def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# convert seconds to min and seconds
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    # model and parameter initialization
    OUTPUT_DIM = 10
    model = AlexNet(OUTPUT_DIM)
    model.apply(initialize_parameters)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    FOUND_LR = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=FOUND_LR)

    # train loop variables
    EPOCHS = 2
    best_valid_loss = float('inf')
    best_model_path = None

    # Start MLflow run
    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id

        # Log hyperparameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", FOUND_LR)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("batch_size", train_iterator.batch_size)

        # iterate over epochs
        for epoch in trange(EPOCHS, desc="Epochs"):
            start_time = time.monotonic()

            # train and validate
            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

            # Log metrics for this epoch
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)
            mlflow.log_metric("valid_acc", valid_acc, step=epoch)

            # Save best model checkpoint
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_path = os.path.join(ARTIFACT_DIR, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)

            # print progress
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%")

        # after training ensure checkpoint exists
        if best_model_path is None:
            raise RuntimeError("Training did not produce a best model checkpoint")

        # load best model for evaluation
        model.load_state_dict(torch.load(best_model_path))
        model.to(device).eval()

        # prepare input example for mlflow signature
        sample_input, _ = next(iter(valid_iterator))
        input_example = sample_input[0:1].cpu().numpy()
        with torch.no_grad():
            output_example = model(sample_input.to(device))[0][0:1].cpu().numpy()
        signature = infer_signature(input_example, output_example)

        # Log final model
        mlflow.pytorch.log_model(
            model,
            artifact_path="models/challenger",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            conda_env=CONDA_ENV_FILE,
        )

        # alias version as challenger
        client = MlflowClient()
        filter_str = f"name = '{REGISTERED_MODEL_NAME}' and run_id = '{run_id}'"
        version = client.search_model_versions(filter_str)[0].version
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=CHALLENGER_ALIAS,
            version=version)
        print(f"Logged and aliased version {version} as {CHALLENGER_ALIAS}.")

        # evaluation on test set
        test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

        # Log test metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)


if __name__ == "__main__":
    main()
