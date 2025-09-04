import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot

import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm


def train_one_epoch(train_dataloader, model, optimizer, loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training", ncols=80):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

        train_loss += (loss_value.item() - train_loss) / (batch_idx + 1)
        preds = output.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {accuracy:.2f}%")

    return float(train_loss), float(accuracy)


def valid_one_epoch(valid_dataloader, model, loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc="Validating", ncols=80):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss_value = loss(output, target)

            valid_loss += (loss_value.item() - valid_loss) / (batch_idx + 1)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"Validation Loss: {valid_loss:.4f} | Validation Accuracy: {accuracy:.2f}%")

    return float(valid_loss), float(accuracy)


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    valid_loss_min = None
    logs = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot()])
    else:
        liveloss = None

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        valid_loss, valid_acc = valid_one_epoch(data_loaders["valid"], model, loss)

        print(
            f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc:.2f}%"
        )

        if valid_loss_min is None or valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print(f"New minimum validation loss: {valid_loss:.4f}. Model saved.")
            valid_loss_min = valid_loss

        scheduler.step(valid_loss)

        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["accuracy"] = train_acc
            logs["val_accuracy"] = valid_acc
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing", ncols=80):
            data, target = data.to(device), target.to(device)

            logits = model(data)
            loss_value = loss(logits, target)

            test_loss += (loss_value.item() - test_loss) / (batch_idx + 1)
            preds = logits.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return test_loss, accuracy

    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        train_loss_value = lt[0]
        train_accuracy_value = lt[1]
        assert not np.isnan(lt[0]), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        valid_loss_value = lv[0]
        valid_accuracy_value = lv[1]
        
        assert not np.isnan(valid_loss_value), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects
    tv = one_epoch_test(data_loaders["test"], model, loss)
    test_loss_value = tv[0]
    test_accuracy_value = tv[1]

    assert not np.isnan(test_loss_value), "Test loss is nan"
