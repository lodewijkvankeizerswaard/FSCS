import torch
import torch.nn as nn
import torchvision.models as models

ADULT_DATASET_FEATURE_SIZE = 105
NODE_SIZE = 80


def get_featurizer(dataset_name: str):
    """
    Returns the model architecture for the provided dataset_name. 
    """
    if dataset_name == 'adult':
        model = AdultFeaturizer()
        out_features = NODE_SIZE

    elif dataset_name == 'celeba':
        model = CelebAFeaturizer()
        out_features = 2048

    elif dataset_name == 'civil':
        model = CivilFeaturizer()
        out_features = 80

    elif dataset_name == 'chexpert':
        model = CheXPertFeaturizer()
        out_features = 1024
    else:
        assert False, f'Unknown network architecture \"{dataset_name}\"'

    return out_features, model

def rename_attribute(obj, old_name, new_name):
    obj._modules[new_name] = obj._modules.pop(old_name)

def drop_classification_layer(model):
    return torch.nn.Sequential(*(list(model.children())[:-1]))


class AdultFeaturizer(nn.Module):
    def __init__(self):
        super(AdultFeaturizer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(ADULT_DATASET_FEATURE_SIZE, NODE_SIZE),
            nn.SELU()
        )

    def forward(self, x):
        output = self.model(x)
        return output


class CelebAFeaturizer(nn.Module):
    def __init__(self):
        super(CelebAFeaturizer, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = drop_classification_layer(self.model)

    def forward(self, x):
        return self.model(x)


class CivilFeaturizer(nn.Module):
    def __init__(self):
        super(CivilFeaturizer, self).__init__()
        bert = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased', return_dict=False)    # Download model and configuration from S3 and cache

        for param in bert.parameters():
            param.requires_grad = False

        bert.classifier = nn.Sequential(
            nn.Linear(768, NODE_SIZE),
            nn.SELU()   
        )
        self.bert = bert

    def forward(self, x):
        output = self.bert(**x)[0]
        return output


class CheXPertFeaturizer(nn.Module):
    def __init__(self):
        super(CheXPertFeaturizer, self).__init__()
        model = models.densenet121(pretrained=True)
        model = drop_classification_layer(model)
        self.model = nn.Sequential(model, nn.AvgPool2d((1, 1)))

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # print(AdultFeaturizer())
    # print(CelebAFeaturizer())
    # print(CivilFeaturizer())
    # CheXPertFeaturizer()
    pass