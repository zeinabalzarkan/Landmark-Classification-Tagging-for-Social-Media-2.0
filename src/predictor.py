import os
import torch.nn as nn
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as T
from .helpers import get_data_location
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import resize, center_crop
from torchvision.transforms.functional import resize, center_crop, convert_image_dtype, normalize
from typing import List
from torchvision.transforms.functional import normalize

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class Predictor(nn.Module):
    def __init__(self, model, class_names, mean, std):
        super().__init__()
        self.model = model.eval()
        self.class_names = class_names
        self.mean = mean.tolist()
        self.std = std.tolist()

    def forward(self, x):
        # Forward pass through the model
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input tensor x
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        return (x - mean) / std

def predictor_test(test_dataloader, model_reloaded):
    """
    Evaluate the model on the test dataset and compute accuracy.
    Assumes that the dataloader provides raw images, so normalization
    is applied here to match training conditions.
    """
    model_reloaded.eval()

    pred = []
    truth = []

    for x_batch, label_batch in tqdm(test_dataloader, leave=True, ncols=80):
        with torch.no_grad():
            # Normalize the batch if needed
            # Uncomment the following line if your model expects normalized inputs
            # x_batch = model_reloaded.normalize(x_batch)

            outputs = model_reloaded(x_batch)
            preds = outputs.argmax(dim=1)

        pred.extend(preds.cpu().numpy())
        truth.extend(label_batch.cpu().numpy())

    pred = np.array(pred)
    truth = np.array(truth)

    accuracy = (pred == truth).mean()
    print(f"Accuracy: {accuracy:.4f}")

    return truth, pred

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    from .model import MyModel
    from .helpers import compute_mean_and_std

    mean, std = compute_mean_and_std()

    model = MyModel(num_classes=3, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    predictor = Predictor(model, class_names=['a', 'b', 'c'], mean=mean, std=std)

    out = predictor(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 3]
    ), f"Expected an output tensor of size (2, 3), got {out.shape}"

    assert torch.isclose(
        out[0].sum(),
        torch.Tensor([1]).squeeze()
    ), "The output of the .forward method should be a softmax vector with sum = 1"
