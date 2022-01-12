###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import numpy as np

ADULT_DATASET_FEATURE_SIZE = 14
NODE_SIZE = 80

def set_seed(seed):
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

def drop_classification_layer(model):
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def get_model(dataset_name: str, pretrained: bool=True):
    """
    Returns the model architecture for the provided dataset_name. 
    """
    # TODO:
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
        out_features = 1024
        
    else:
        assert False, f'Unknown network architecture \"{dataset_name}\"'
        
    return out_features, model

def split(x: torch.Tensor, d: torch.Tensor):
    # Groups x samples based on d-values
    sorter = torch.argsort(d)
    _, counts = torch.unique(d, return_counts=True)
    return sorter, torch.split(x[sorter], counts.tolist())


class FairClassifier(nn.Module):
    def __init__(self, input_model: str):
        """
        FairClassifier Model
        """
        super(FairClassifier, self).__init__()
        in_features, self.featurizer = get_model(input_model)

        # Fully Connected models for binary classes
        self.fc0 = nn.Linear(in_features, 1)
        self.fc1 = nn.Linear(in_features, 1)

        # Join Classifier T
        self.joint_classifier = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor, d_true: torch.Tensor, d_random: torch.Tensor):
        """
        Forward Pass
        """
        features = self.featurizer(x)

        joint_y = self.joint_classifier(features)

        # Split on random sampled d-values
        random_indices, random_d_0, random_d_1 = split(features, d_random)

        # Split on true d-values
        group_indices, group_d_0, group_d_1 = split(features, d_true)

        group_agnostic_y = torch.zeros(features.shape)
        group_specific_y = torch.zeros(features.shape)

        group_agnostic_y[random_indices] = torch.cat([self.fc0(random_d_0), self.fc1(random_d_1)])
        group_specific_y[group_indices] = torch.cat([self.fc0(group_d_0), self.fc1(group_d_1)])

        return F.sigmoid(joint_y), F.sigmoid(group_specific_y), F.sigmoid(group_agnostic_y)

if __name__ == "__main__":
    # model = FairClassifier('adult')
    # print(model)

    model = get_model('celeba')