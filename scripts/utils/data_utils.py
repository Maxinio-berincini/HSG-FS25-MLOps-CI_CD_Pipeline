import copy
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# load config
from scripts.utils.config import (
    BATCH_SIZE,
    VALID_RATIO,
    SEED,
    DATA_ROOT,
)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


train_data = datasets.CIFAR10(root=DATA_ROOT,
                              train=True,
                              download=True)

means = train_data.data.mean(axis=(0, 1, 2)) / 255
stds = train_data.data.std(axis=(0, 1, 2)) / 255

print(f'Calculated means: {means}')
print(f'Calculated stds: {stds}')

train_transforms = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
])

train_data = datasets.CIFAR10(DATA_ROOT,
                              train=True,
                              download=True,
                              transform=train_transforms)

test_data = datasets.CIFAR10(DATA_ROOT,
                             train=False,
                             download=True,
                             transform=test_transforms)

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data,
                                           [n_train_examples, n_valid_examples])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing  examples: {len(test_data)}')


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)


def estimate_required_samples(epsilon, delta):
    """
    Estimate the number of samples required for (epsilon, delta) gurantees in significance testing.
    epison: error tollerance
    delta: 1-confidence level
    """

    ### using Hoeffding's inequality
    return int(np.ceil((np.log(2/delta)) / (2 * epsilon**2)))


def calculate_achievable_confidence(test_set_size, epsilon):
   
    """
    Calculate the achievable confidence level based on the test set size and epsilon.
    """
    delta = 2 * np.exp(-2 * test_set_size * epsilon**2)
    confidence = 1 - delta

    return min(max(confidence, 0.0), 1.0)  # Ensure confidence is between 0 and 1


