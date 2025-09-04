import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing

from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import random
import numpy as np

from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = 1, limit: int = -1
):
    """
    Returns dictionary of train/valid/test data loaders
    """
    seed_everything()

    data_loaders = {"train": None, "valid": None, "test": None}
    base_path = Path(get_data_location())

    # Compute mean and std of dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # Transforms
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }

    # Load full dataset from train folder
    full_dataset = datasets.ImageFolder(base_path / "train")

    n_total = len(full_dataset)
    indices = list(range(n_total))

    if limit > 0:
        indices = indices[:limit]
        n_total = limit

    split = int(valid_size * n_total)
    random.shuffle(indices)
    valid_idx, train_idx = indices[:split], indices[split:]

    # Apply transform using Subset
    train_dataset = datasets.ImageFolder(base_path / "train", transform=data_transforms["train"])
    valid_dataset = datasets.ImageFolder(base_path / "train", transform=data_transforms["valid"])

    # Use samplers to enforce split
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # Create loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    # Test set
    test_data = datasets.ImageFolder(base_path / "test", transform=data_transforms["test"])
    if limit > 0:
        test_indices = list(range(min(limit, len(test_data))))
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch from train loader.
    """
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
        transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
    ])
    images = invTrans(images).clamp(0, 1)

    class_names = data_loaders["train"].dataset.classes
    images = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    fig = plt.figure(figsize=(25, 4))
    for idx in range(min(max_n, len(images))):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):

    assert set(data_loaders.keys()) == {"train", "valid", "test"}, "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you " \
                                       "forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert (
        len(labels) == 2
    ), f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):

    visualize_one_batch(data_loaders, max_n=2)
