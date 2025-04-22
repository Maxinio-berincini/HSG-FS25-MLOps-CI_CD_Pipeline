import time
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.notebook import tqdm, trange
import mlflow
import mlflow.pytorch
import logging

# MLflow setup
from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("alexnet_experiment")

from data_utils import train_iterator, valid_iterator, test_iterator
from model import AlexNet, count_parameters, initialize_parameters
import torch.nn as nn

os.makedirs("model_artifacts", exist_ok=True)

logging.getLogger("mlflow").setLevel(logging.DEBUG)

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

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):
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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    OUTPUT_DIM = 10
    model = AlexNet(OUTPUT_DIM)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.apply(initialize_parameters)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    FOUND_LR = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=FOUND_LR)

    EPOCHS = 25
    best_valid_loss = float('inf')

    # Start MLflow run
    with mlflow.start_run(run_name="alexnet_run"):

        print("ARTIFACT URI:", mlflow.get_artifact_uri())

        # Log hyperparameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", FOUND_LR)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("batch_size", train_iterator.batch_size)

        for epoch in trange(EPOCHS, desc="Epochs"):
            start_time = time.monotonic()

            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

            # Log metrics for this epoch
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)
            mlflow.log_metric("valid_acc", valid_acc, step=epoch)

            # Save & log best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                # torch.save(model.state_dict(), "model_artifacts/model.pt")
                # mlflow.log_artifact("model_artifacts/model.pt", artifact_path="models/best")

                model_path = "model_artifacts/model.pt"
                torch.save(model.state_dict(), model_path)
                # then upload that one file
                mlflow.log_artifact(model_path, artifact_path="models/best")

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%")

        # Evaluate on test set
        model.load_state_dict(torch.load('model_artifacts/model.pt'))
        test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

        # Log test metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)

        # Log the final model
        mlflow.pytorch.log_model(model, artifact_path="models/final")


if __name__ == "__main__":
    main()
