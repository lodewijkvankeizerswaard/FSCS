import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from model import *
from train_model import *
from data import *

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

def evaluation_statistics(data, tau, model):
    """Calculates the coverage based on the threshold tau."""
    correct, incorrect = 0, 0, 0
    
    for batch, label in data:
        prediction = model(batch)
        if margin(batch, prediction, label) >= tau: 
            correct += 1
        elif margin(batch, prediction, label) <= -tau:
            incorrect += 1
        # elif :
        #     fp += 1
    
    accuracy = correct / (correct + incorrect)
    # precision = correct / (correct + fp)
    coverage = (correct + incorrect) / len(data)
    
    return accuracy, coverage

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

def accuracy_coverage_plot(data, taus, model):
    accuracy, coverage = [], []
    for tau in taus:
        acc, cov = evaluation_statistics(data, tau, model)
        accuracy.append(acc)
        coverage.append(cov)
    
    plt.plot(accuracy, coverage)
    plt.xlabel('coverage')
    plt.ylabel('accuacy')
    plt.show()
    
    return area_under_curve(accuracy, coverage)

def precision_coverage_plot(data, taus, model):
    precision, coverage = [], []
    for tau in taus:
        _, prec, cov = evaluation_statistics(data, tau, model)
        precision.append(prec)
        coverage.append(cov)
    
    plt.plot(precision, coverage)
    plt.xlabel('coverage')
    plt.ylabel('precision')
    plt.show()

# def PDF(M):
#     """Estimates the probability density function based on the margin values."""
#     #TODO: tau moet als input.
    
#     M = np.sort(M)
#     plt.hist(M, bins=3, density=True)
#     plt.plot(M, norm.pdf(M))
#     plt.show()

# def CDF(M):
#     #TODO: tau als input. 
#     M = np.sort(M)
#     return norm.cdf(M)

# def condCDF(tau):
#     pass

# def selective_accuracy(tau):
#     """Takes the threshold tau as input and returns the selective accuracy."""
#     return (1 - CDF(tau)) / (CDF(-tau) + 1 - CDF(tau))

# def selective_precision(tau):
#     """Takes the threshold tau as input and returns the selective precision.""" 
#     return (1 - condCDF(tau)) / (condCDF(-tau) + 1 - condCDF(tau))

def test_model(model: nn.Module, dataset: str, batch_size: int, device: torch.device, seed: int, dataset_root: str, progress_bar: bool):
    """
    Tests a trained model on the test set.
    Args:
        model: Model architecture to test.
        dataset: Specify dataset where test_set is loaded from.
        batch_size: Batch size to use in the test.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: The average accuracy on the test set (independent of the attribute).
    """

    set_seed(seed)
    test_set = get_test_set(dataset, root=dataset_root)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                shuffle=True, num_workers=3)

    num_correct = 0
    total_samples = 0 
    with torch.no_grad():
        for x, t, _ in tqdm(test_loader, desc="test", disable=progress_bar):
            x = x.to(device)
            t = t.to(device).squeeze()

            pred_joint, _, _ = model.forward(x)
            
            num_correct += num_correct_predictions(pred_joint, t)
            total_samples += len(x)

    avg_accuracy = num_correct / total_samples
    
    return avg_accuracy

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    batch_size = 32
    seed = 42
    dataset_root = "data"

    dataset = 'adult'
    checkpoint_name = dataset + '.pt'
    checkpont_path = os.path.join("models", checkpoint_name)
    
    model = get_model(dataset)
    #model = FairClassifier(dataset, nr_attr_values=len(ADULT_ATTRIBUTE['values']))
    model = model.load_state_dict(torch.load(checkpont_path))
    
    progress_bar = False
    test_model(model, dataset, batch_size, device, seed, dataset_root, progress_bar)
    
    #accuracy_coverage_plot(data, taus, model)