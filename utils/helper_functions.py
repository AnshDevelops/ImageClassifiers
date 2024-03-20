import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import torch


def get_num_files(path):
    """
    Walks through a directory and prints its contents
    :param path: Path to directory.
    """

    for path, _, filenames in os.walk(path):
        print(f"There are {len(filenames)} images in '{path}'.")


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folder names in a target directory.
    Assumes target directory is in standard image classification format.

    :param directory: target directory to load classnames from.
    :return: Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    """

    # get class names by scanning target directory
    classes = sorted(entry.name for entry in os.scandir(directory))
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}")

    # create a dictionary of index labels (computers prefer numerical labels over string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots loss curves given a results dictionary of a model

    Args:
        results (dict): dictionary containing list of values, e.g.
        {"train_loss": [...],
        "train_acc": [...],
        "train_loss": [...],
        "train_loss": [...]}
    """

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.xlabel("Epochs")
    plt.legend()
