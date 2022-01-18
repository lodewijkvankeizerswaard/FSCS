from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from featurizers import get_featurizer

import numpy as np

class FairClassifier(nn.Module):
    def __init__(self, input_model: str, nr_attr_values: int = 2):
        """
        FairClassifier Model
        """
        super(FairClassifier, self).__init__()
        in_features, self.featurizer = get_model(input_model)

        # Fully Connected models for binary classes
        self.group_specific_models = nn.ModuleList([nn.Linear(in_features, 1) for key in range(nr_attr_values)])

        # Join Classifier T
        self.joint_classifier = nn.Linear(in_features, 1)

    def split(self, x: torch.Tensor, d: torch.Tensor):
        # Groups x samples based on d-values
        sorter = torch.argsort(d, dim=1)
        _, counts = torch.unique(d, return_counts=True)
        return sorter, torch.split(torch.squeeze(x[sorter, :]), counts.tolist())

    def group_forward(self, x: torch.Tensor, d: torch.Tensor):
        """ The forward pass of the group specific models. """

        features = self.featurizer(x).squeeze()

        # Sort the inputs on the value of d
        group_indices, group_splits = self.split(features, d)

        group_pred = torch.zeros(features.shape[0], device=self.device())
        group_pred[group_indices] = torch.cat([self.group_specific_models[i](group_split) for i, group_split in enumerate(group_splits)])

        return torch.sigmoid(group_pred)

    def forward(self, x: torch.Tensor, d: torch.Tensor = None, d_tilde: torch.Tensor = None):
        """Returns the model prediction by the joint classifier, and the group specific and 
        group agnostic models if `d` and `d_tilde` are given respectively.

        Args:
            x (torch.Tensor): the input values for the classifer
            d (torch.Tensor): the true attributes of the data points
            d_tilde (torch.Tensor): a random attributes

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): the label prediction by the joint classifer
            the group specific model, and the group agnostic model. If `d` or `d_tilde` are `None`, the
            respective output is also `None`.
        """

        features = self.featurizer(x).squeeze()

        # Group specific
        if type(d) == torch.Tensor:
            group_spe_indices, group_spe_splits = self.split(features, d)

            group_spe_pred = torch.zeros(features.shape[0], device=self.device())
            group_spe_pred[group_spe_indices] = torch.cat([self.group_specific_models[i](group_split) for i, group_split in enumerate(group_spe_splits)])
            group_spe_pred = torch.sigmoid(group_spe_pred)
        else:
            group_spe_pred = None

        # Group agnostic
        if type(d_tilde) == torch.Tensor:
            group_agn_indices, group_agn_splits = self.split(features, d_tilde)

            group_agn_pred = torch.zeros(features.shape[0], device=self.device())
            group_agn_pred[group_agn_indices] = torch.cat([self.group_specific_models[i](group_split) for i, group_split in enumerate(group_agn_splits)])
            group_agn_pred = torch.sigmoid(group_agn_pred)
        else:
            group_agn_pred = None

        joint_pred = self.joint_classifier(features)
        joint_pred = torch.sigmoid(joint_pred)

        return joint_pred.squeeze(), group_spe_pred, group_agn_pred

    def device(self):
        return next(self.parameters()).device
