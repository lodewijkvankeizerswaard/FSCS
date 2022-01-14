import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from sklearn.metrics import auc
from scipy.stats import norm
from numpy.random import normal

#### TODO: Tau as input ipv M, bij selective accuracy

def softmax(x):
    """Takes the data as input and returns the softmax response."""
    b = max(x) 
    return (math.e**(x - b)) / sum(math.e**(x - b))

def confidence_score(x):
    """Takes the data as input and returns its confidence score."""
    s = softmax(x)
    return 0.5 * math.log(s / (1 - s))

def margin(x, prediction, label):
    """Takes the data, its prediction and its label as input and returns the 
    value for its margin."""
    if prediction == label:
        return confidence_score(x)
    else:
        return -confidence_score(x)

def plot_margin(M):
    """Takes the margin values as input and plots its histogram."""
    bins = int(np.sqrt(len(M)))
    plt.hist(M, bins=bins)
    plt.show()

def PDF(M):
    """Estimates the probability density function based on the margin values."""
    #TODO: tau moet als input.
    
    M = np.sort(M)
    plt.hist(M, bins=3, density=True)
    plt.plot(M, norm.pdf(M))
    plt.show()

def CDF(M):
    #TODO: tau als input. 
    M = np.sort(M)
    return norm.cdf(M)

def condCDF(tau):
    pass

def selective_accuracy(tau):
    """Takes the threshold tau as input and returns the selective accuracy."""
    return (1 - CDF(tau)) / (CDF(-tau) + 1 - CDF(tau))

def selective_precision(tau):
    """Takes the threshold tau as input and returns the selective precision.""" 
    return (1 - condCDF(tau)) / (condCDF(-tau) + 1 - condCDF(tau))

def calc_coverage(data, tau, model):
    """Calculates the coverage based on the threshold tau."""
    coverage = 0
    
    for batch, label in data:
        prediction = model(batch)
        if margin(batch, prediction, label) >= tau or margin(batch, prediction, label) <= -tau:
            coverage += 1
    
    return coverage/len(data)

def accuracy_coverage_plot():
    pass

def area_under_curve(x, y):
    """Takes lists of x and y coordinates as input and calculates the area under
    the curve."""
    return auc(x, y)

def average_acc_cov(area1, area2):
    """Takes the area under two curves as input and returns its average.1"""
    return (area1 + area2).mean()

def area_between_curves(area1, area2):
    """Takes the areas under two curves as input and calculates the area between
    the two curves."""
    return abs(area1 - area2)
