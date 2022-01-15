from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import numpy as np

ADULT_DATASET_FEATURE_SIZE = 105
NODE_SIZE = 80


def drop_classification_layer(model):
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def get_model(dataset_name: str, pretrained: bool=True):
    """
    Returns the model architecture for the provided dataset_name. 
    """
    if dataset_name == 'adult':
        model = nn.Sequential(
            nn.Linear(ADULT_DATASET_FEATURE_SIZE, NODE_SIZE),
            nn.SELU()
            )
        out_features = NODE_SIZE

    elif dataset_name == 'celeba':
        model = models.resnet50(pretrained=pretrained)
        model = drop_classification_layer(model)
        out_features = 2048

    elif dataset_name == 'civilcomments':
        bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        fc_model = nn.Sequential(
            nn.Linear(1024, NODE_SIZE),
            nn.SELU()
            )
        model = torch.nn.Sequential(bert_model, fc_model)
        out_features = NODE_SIZE

    elif dataset_name =='chexpert':
        model = models.densenet121(pretrained=pretrained)
        model = drop_classification_layer(model)
        model = nn.Sequential(model, nn.AdaptiveAvgPool2d((1,1)))
        out_features = 1024
        
    else:
        assert False, f'Unknown network architecture \"{dataset_name}\"'
        
    return out_features, model


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