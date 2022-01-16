import torch
from torch import nn
import numpy as np

import os
from torch.nn.modules import loss
from tqdm import tqdm
import argparse

from data import ADULT_ATTRIBUTE, get_train_validation_set, get_test_set
from model import FairClassifier
from torch.utils.tensorboard import SummaryWriter

LAMBDA = 0.7 

def set_seed(seed: int):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

def train_model(model: nn.Module, dataset: str, lr: float, batch_size: int, 
                epochs: int, checkpoint_name: str, device: torch.device, dataset_root:str, progress_bar: bool):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        dataset: Specifiec dataset.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    writer = SummaryWriter()

    # Load the datasets
    train_set, val_set = get_train_validation_set(dataset, root=dataset_root)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)
    # validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
    #                                                 shuffle=True, num_workers=2)

    # Initialize the optimizer and loss function
    group_specific_params = [param for gsm in model.group_specific_models for param in gsm.parameters()]
    feature_extractor_params = model.featurizer.parameters()
    joint_classifier_params = model.joint_classifier.parameters()

    group_specific_optimizer = torch.optim.SGD(group_specific_params, lr=lr)
    feature_extractor_optimizer = torch.optim.SGD(feature_extractor_params, lr=lr)
    joint_classifier_optimizer = torch.optim.SGD(joint_classifier_params, lr=lr)

    loss_module = nn.BCELoss()

    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    best_accuracy = 0
    for epoch in tqdm(range(epochs), position=0, desc="epoch", disable=progress_bar):

        # Group specific training
        group_correct, group_total = 0, 0
        for x, t, d in tqdm(train_loader, position=1, desc="group", leave=False, disable=progress_bar):
            x = x.to(device)
            t = t.to(device)
            d = d.to(device)
            group_specific_optimizer.zero_grad()
            pred_group_spe = model.group_forward(x, d)

            group_specific_loss = loss_module(pred_group_spe, t.squeeze())
            group_specific_loss.backward()

            group_specific_optimizer.step()

            group_correct += num_correct_predictions(pred_group_spe, t)
            group_total += len(x)

        writer.add_scalar(checkpoint_name + "/group_acc", group_correct / group_total, epoch)

        # Feature extractor and joint classifier trainer
        joint_correct, joint_total = 0, 0
        for x, t, d in tqdm(train_loader, position=1, desc="joint", leave=False, disable=progress_bar):
            x = x.to(device)
            t = t.to(device).squeeze()
            d = d.to(device)
            # Sample d values for the group agnostic model
            d_tilde = train_loader.dataset.sample_d(d.shape)

            # Get model predictions
            pred_joint, pred_group_spe, pred_group_agn = model.forward(x, d, d_tilde)

            # Calculate L_0 and L_R
            L_0 = loss_module(pred_joint, t)
            L_R = loss_module(pred_group_spe, t) - loss_module(pred_group_agn, t)

            # Add L_R to the feature extractor gradients (but not to the joint classifier)
            feature_extractor_optimizer.zero_grad()
            L_R.backward(retain_graph=True)
            joint_classifier_optimizer.zero_grad()

            # Add L_0 to both the feature extractor and joint classifier gradients
            L_0.backward()

            # Update the classifier and feature extractor
            joint_classifier_optimizer.step()
            feature_extractor_optimizer.step()
            
            joint_correct += num_correct_predictions(pred_joint, t)
            joint_total += len(x)

        writer.add_scalar(checkpoint_name + "/joint_acc", joint_correct / joint_total, epoch)
    
    writer.close()
        
    # Load best model and return it.
    # model.load_state_dict(torch.load("models/" + checkpoint_name))
    torch.save(model.state_dict(), os.path.join("models", checkpoint_name))

    return model

def num_correct_predictions(predictions: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = (predictions > 0.5).long()
    count = (predictions == targets.squeeze()).sum()
    return count.item()

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
                                                shuffle=True, num_workers=4)

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

def main(dataset: str, lr: float, batch_size: int, epochs: int, seed: int, dataset_root:str, progress_bar: bool):
    """
    Function that summarizes the training and testing of a model.

    Args:
        dataset: Dataset to test.
        batch_size: Batch size to use in the test.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.
    """

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(seed)

    checkpoint_name = dataset+ '.pt'
    checkpont_path = os.path.join("models", checkpoint_name)
    model = FairClassifier(dataset, nr_attr_values=len(ADULT_ATTRIBUTE['values'])).to(device)
    if os.path.exists(checkpont_path):
        model.load_state_dict(torch.load(checkpont_path))
    else:
        model = train_model(model, dataset, lr, batch_size, epochs,
                            checkpoint_name, device, dataset_root, progress_bar)
    test_results = test_model(model, dataset, batch_size, device, seed, dataset_root, progress_bar)
    print("Accuracy on the test set:", test_results)
    return test_results

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--dataset', default='adult', type=str,
                        help='Name of the dataset to evaluate on.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Minibatch size.')

    # Other hyperparameters
    parser.add_argument('--epochs', default=20, type=int,
                        help='Max number of epochs.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results.')

    # Other arguments
    parser.add_argument('--dataset_root', default="data", type=str,
                        help="the root of the data folders.")
    parser.add_argument('--progress_bar', action="store_false",
                        help="Turn progress bar on.")

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)
        