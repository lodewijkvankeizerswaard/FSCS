import matplotlib
from matplotlib import rc
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# rc('text', usetex=True)

def confidence_score(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * np.log(x / (1 - x))

def split(x: torch.Tensor, d: torch.Tensor):
    # Groups x samples based on d-values
    sorter = torch.argsort(d, dim=0)
    _, counts = torch.unique(d, return_counts=True)
    return torch.split(x[sorter, :], counts.tolist()), torch.split(d[sorter], counts.tolist())

def margin(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compares the prediction and the target for the calculation of the margin 
    for a datapoint.
    Args:
        prediction: The predictions of the samples.
        target: The corresponding correct targets for the prediction.
    Returns:
        margin: The margin values for the corresponding samples.
    """
    pred = prediction.clone()
    correct = (torch.round(pred) == target).type(torch.int) * 2 - 1
    pred[pred<0.5] = 1 - pred[pred<0.5]
    margin = correct * confidence_score(pred)
    # margin[margin >= 5] = 5
    # margin[margin <= -5] = -5
    return margin

def margin_group(predictictions: torch.Tensor, targets: torch.Tensor, attributes: torch.Tensor) -> dict:
    """This function splits the predictions and targets on group assignment and calculates the corresponding
        group specific margin. """
    pred_split, d_split = split(predictictions, attributes)
    tar_split, _ = split(targets, attributes)
    margins = {int(d[0]):margin(p, t) for p, t, d in zip(pred_split, tar_split, d_split)}
    return margins

def precision_group(predictictions: torch.Tensor, targets: torch.Tensor, attributes: torch.Tensor) -> dict:
    """Calculate the group specific precisions."""
    pred_split, d_split = split(predictictions, attributes)
    tar_split, _ = split(targets, attributes)
    margin_precision = {}
    for group_pred, group_tar, group_attr in zip(pred_split, tar_split, d_split):
        pred_round = torch.round(group_pred)
        zero_index = (pred_round == 0).nonzero()[:,0]
        group_pred_y_hat_1 = group_pred[~zero_index,:]
        group_tar_y_hat_1 = group_tar[~zero_index,:]
        margin_precision[group_attr[0].item()] = margin(group_pred_y_hat_1, group_tar_y_hat_1)
    return margin_precision

def evaluation_statistics(predictions: torch.Tensor, targets: torch.Tensor, attributes: torch.Tensor):
    """
    Computes the evaluation statistics for the test data.
    Args:
        predictions: The predictions of the samples.
        targets: The corresponding targets for the predictions.
        attributes: The corresponding attributes for the predictiosn.
    Returns:
        area_under_curve: The area under the accuracy-coverage curve.
        area_between_curves_val: The area between the precision-coverage curves.
        M_group: The margin values of the samples per group.
        A_group: The accuracies for different values of tau per group.
        C_group: The corresponding coverages for different values of tau per group.
        P_M_group: The margin values of the samples 
        P_A_group: The precision values for different values of tau per group.
        P_C_group: The corresponding coverages for different values of tau per group.
        """
    M = margin(predictions, targets) 
    max_tau = torch.max(torch.abs(M)).item()
    taus = np.linspace(0, max_tau, num=2000)

    # Compute overal margin and AUC statistics
    CDF = lambda margin, tau: (len(margin[margin <= tau]) / len(margin))
    CDF_correct = lambda margin, tau: 1 - CDF(margin, tau)
    CDF_covered = lambda margin, tau: CDF(margin, -tau) + 1 - CDF(margin, tau)

    A = [CDF_correct(M, tau) / CDF_covered(M, tau) if CDF_covered(M, tau) > 0 else 1 for tau in taus]
    C = [CDF_covered(M, tau) for tau in taus]
    area_under_curve = auc(C, A)

    # Compute group specific margins and accuracies
    M_group = margin_group(predictions, targets, attributes)
    A_group = {group_key: [CDF_correct(group_margin, tau) / CDF_covered(group_margin, tau) if CDF_covered(group_margin, tau) > 0 else 1 for tau in taus] for group_key, group_margin in M_group.items()}
    C_group = {group_key: [CDF_covered(group_margin, tau) for tau in taus] for group_key, group_margin in M_group.items()}

    # Compute the group specific precisions for Y_hat = 1
    P_M_group = precision_group(predictions, targets, attributes)
    P_A_group = {group_key: [CDF_correct(group_margin, tau) / CDF_covered(group_margin, tau) if CDF_covered(group_margin, tau) > 0 else 1 for tau in taus] for group_key, group_margin in P_M_group.items()}
    P_C_group = {group_key: [CDF_covered(group_margin, tau) for tau in taus] for group_key, group_margin in P_M_group.items()}

    # area_under_curve_group_precision = [auc(P_C_group[group], P_A_group[group]) for group in P_M_group.keys()]
    # area_between_curves_val = area_between_curves(area_under_curve_group_precision[0], area_under_curve_group_precision[1])
    area_between_curves = abc(P_A_group)
    
    return area_under_curve, area_between_curves, M_group, A_group, C_group, P_M_group, P_A_group, P_C_group

def plot_margin_group(margins: dict) -> matplotlib.figure.Figure:
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
    ax.set_xlabel('k (x)')
    ax.legend(loc="upper left")
    return fig

# def area_between_curves(area1: float, area2: float) -> float:
#     return abs(area1 - area2)

def abc(precisions: dict) -> float:
    """
    Calculates the area between two curves.
    Args:
        precisions: The precision values for the two curves.
    Returns:
        area: The area between the two curves.
    """
    n = len(precisions[0])
    xrange = np.arange(0, n, step=1/n)
    area = 0
    for i in range(2, n):
        area += abs(auc(precisions[0][i-2:i], xrange[i-2:i]) - auc(precisions[1][i-2:i], xrange[i-2:i]))
    return area

def accuracy_coverage_plot(accuracies: dict, coverages: dict, ylabel: str) -> matplotlib.figure.Figure:
    """
    Plots the accuracy vs. the coverage.
    Args:
        accuracies: Dict of accuracies split on group/attribute and depending on different values of tau.
        coverages: The corresponding coverages for the accuracies.
    """
    fig = plt.figure()
    for group in accuracies.keys():
        coverages[group].reverse()
        accuracies[group].reverse()
        plt.plot(coverages[group], accuracies[group], label="Group " + str(int(group)))
    plt.xlabel('coverage')
    plt.ylabel(ylabel)
    plt.ylim([0.4, 1.01])
    plt.xlim([0.15, 1.0])
    plt.legend(loc="lower left")
    plt.title("Group-specific "+ ylabel+"-coverage curves.")
    return fig

