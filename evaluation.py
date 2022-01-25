import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def confidence_score(x):
    return 0.5 * np.log(x / (1 - x))

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
    if prediction == target:
        margin = confidence_score(prediction) 
    else:
        margin = -confidence_score(prediction)
    return margin
       
def plot_margin(M_0: list, M_1: list):
    """
    Plots the margin distributions for two groups.
    Args:
        M_0: The margin values for Group 0.
        M_1: The margin values for Group 1.
    """
    # bins = int(np.sqrt(len(M_0)))
    bins = 1000
    plt.hist(M_0, bins, alpha=0.5, label='Group 0')
    plt.hist(M_1, bins, alpha=0.5, label='Group 1')
    plt.legend(loc="upper left")
    plt.show()

def evaluation_statistics(predictions: torch.Tensor, targets: torch.Tensor, d: torch.Tensor, tau: float):
    """
    Calculates the statistics necessary for evaluation.
    Args:
        data: The test dataset that is evaluated.
        predictions: The predictions made for the test dataset.
        targets: The corresponding correct targets for the predictions.
        d: The group the datapoints corresponds to.
        tau: The threshold for abstaining predictions.
    Returns:
        num_correct: The number of correct classified datapoints.
        num_classified: The total number of classified datapoints.
        true_pos: The amount of true positives.
        false_pos: The amount of false positives.
        M: The margin values for Group 0 and Group 1.
    """
    num_correct_0, num_incorrect_0, num_correct_1, num_incorrect_1 = 0, 0, 0, 0 
    tp_0, fp_0, tp_1, fp_1 = 0, 0, 0, 0
    M_0, M_1 = [], []

    for i in range(len(targets)):
        marg = margin(predictions[i], targets[i])

        # correct predictions
        if marg >= tau and d[i] == 0:
            M_0.append(marg)
            num_correct_0 += 1

        elif marg >= tau and d[i] == 1:
            M_1.append(marg)
            num_correct_1 += 1
            
        # incorrect predictions
        elif marg <= -tau and d[i] == 0:
            M_0.append(marg)   
            num_incorrect_0 += 1  
        elif marg <= -tau and d[i] == 1:
            M_1.append(marg)
            num_incorrect_1 += 1
        
    num_correct = num_correct_0 + num_correct_1
    num_incorrect = num_incorrect_0 + num_incorrect_1
    num_classified = num_correct + num_incorrect
    
    true_pos = [tp_0, tp_1]
    false_pos = [fp_0, fp_1]
    M = [M_0, M_1]
    
    return num_correct, num_classified, true_pos, false_pos, M 

def average_acc_cov(area1: float, area2: float) -> float:
    return (area1 + area2).mean()

def area_between_curves(area1: float, area2: float) -> float:
    return abs(area1 - area2)

def accuracy_coverage_plot(accuracies: list, coverages: list):
    """
    Plots the accuracy vs. the coverage.
    Args:
        accuracies: List of accuracies depending on different values of tau.
        coverages: The corresponding coverages for the accuracies.
    Returns:
        area_under_curve: The area under the accuracy-coverage curve.
    """
    accuracies.reverse()
    coverages.reverse()

    plt.plot(coverages, accuracies)
    plt.xlabel('coverage')
    plt.ylabel('accuracy')
    plt.ylim([0.4, 1.01])
    plt.xlim([0.15, 1.0])
    plt.show()
    return auc(coverages, accuracies)

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
