import matplotlib
from matplotlib import rc
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

rc('text', usetex=True)

def confidence_score(x):
    return 0.5 * np.log(x / (1 - x))

def split(x: torch.Tensor, d: torch.Tensor):
    # Groups x samples based on d-values
    sorter = torch.argsort(d, dim=0)
    _, counts = torch.unique(d, return_counts=True)
    return torch.split(x[sorter, :].squeeze(dim=1), counts.tolist()), torch.split(d[sorter, :].squeeze(dim=1), counts.tolist())

def margin(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compares the prediction and the target for the calculation of the margin 
    for a datapoint.
    Args:
        prediction: the prediction of a specific datapoint.
        target: the corresponding correct target for the prediction.
    Returns:
        margin: The margin value for the corresponding datapoint.
    """
    correct = (torch.round(prediction) == target).type(torch.int) * 2 - 1
    prediction[prediction<0.5] = 1 - prediction[prediction<0.5]
    return correct * confidence_score(prediction)

def margin_group(predictictions: torch.Tensor, targets: torch.Tensor, attributes: torch.Tensor) -> list:
    """This function splits the predictions and targets on group assignment and calculates the corresponding
        group specific margin. """
    pred_split, d_split = split(predictictions, attributes)
    tar_split, _ = split(targets, attributes)
    margins = {int(d[0]):margin(p, t) for p, t, d in zip(pred_split, tar_split, d_split)}
    return margins
       
def plot_margin(margins: dict) -> matplotlib.figure.Figure:
    """
    Plots the margin distributions for two groups.
    Args:
        margins: A dictionary containing the margins for all groups with label `g` and margins `m`.
    Returns: 
        A matplotlib histogram figure with the margin for each group.
    """
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    for g, m in margins.items():
        ax.hist(m.numpy().flatten(), bins='auto', density=True, alpha=0.5, label='Group ' + str(g))
    ax.set_xlabel(r'$\kappa (x)$')

    ax.legend(loc="upper left")
    return fig

def average_acc_cov(area1: float, area2: float) -> float:
    return (area1 + area2).mean()

def area_between_curves(area1: float, area2: float) -> float:
    return abs(area1 - area2)

def accuracy_coverage_plot(accuracies: dict, coverages: dict):
    """
    Plots the accuracy vs. the coverage.
    Args:
        accuracies: Dict of accuracies split on group/attribute and depending on different values of tau.
        coverages: The corresponding coverages for the accuracies.
    Returns:
        area_under_curve: The area under the accuracy-coverage curve.
    """
    # accuracies.reverse()
    # coverages.reverse()
    fig = plt.figure()
    for group in accuracies.keys():
        coverages[group].reverse()
        accuracies[group].reverse()
        plt.plot(coverages[group], accuracies[group], label="Group " + str(group))
    plt.xlabel('coverage')
    plt.ylabel('accuracy')
    plt.ylim([0.4, 1.01])
    plt.xlim([0.15, 1.0])
    plt.legend(loc="lower left")
    plt.show()
    return fig

def precision_coverage_plot(precisions_0: list, precisions_1: list, coverages: list):
    """
    Plots the precision vs. the coverage for Group 0 and Group 1.
    Args:
        precisions_0: The precision values for Group 0.
        precisions_1: The precision values for Group 1.
        coverages: The corresponding coverages for the precisions.
    Returns:
        area_between_curves: The area between the two precision-coverage curves.
    """
    coverages.reverse()
    precisions_0.reverse()
    precisions_1.reverse()

    plt.plot(coverages, precisions_0, label='Group 0')
    plt.plot(coverages, precisions_1, label='Group 1')
    plt.legend(loc='upper left')
    plt.ylim([0.4, 1.01])
    plt.xlim([0.15, 1.0])
    plt.xlabel('coverage')
    plt.ylabel('precision')
    plt.show()

    area_0 = auc(coverages, precisions_0)
    area_1 = auc(coverages, precisions_1)
    return area_between_curves(area_0, area_1)
