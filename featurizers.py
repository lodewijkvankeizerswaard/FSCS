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

    elif dataset_name == 'civilcomments':
        model = CivilCommentsFeaturizer()
        out_features = 80

    elif dataset_name == 'chexpert':
        model = CheXPertFeaturizer()
        out_features = 1024
    else:
        assert False, f'Unknown network architecture \"{dataset_name}\"'

    return out_features, model


def drop_classification_layer(model, n=1):
    return torch.nn.Sequential(*(list(model.children())[:-n]))


class AdultFeaturizer(nn.Module):
    def __init__(self):
        super(AdultFeaturizer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(ADULT_DATASET_FEATURE_SIZE, NODE_SIZE),
            nn.SELU()
        )

    def forward(self, x):
        return self.model(x)


class CelebAFeaturizer(nn.Module):
    def __init__(self):
        super(CelebAFeaturizer, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = drop_classification_layer(self.model, n=2)

    def forward(self, x):
        return self.model(x)


class CivilCommentsFeaturizer(nn.Module):
    def __init__(self):
        super(CivilCommentsFeaturizer, self).__init__()
        # Using the configuration with a model
        config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')
        config.output_attentions = True
        config.output_hidden_states = True
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased', return_dict=False)    # Download model and configuration from S3 and cache
        self.bert = drop_classification_layer(self.bert, n=2)

        # print(self.bert)
        self.fc_model = nn.Sequential(
            nn.Linear(768, NODE_SIZE),
            nn.SELU()
        )

    def forward(self, x):
        idx = (x==102).nonzero(as_tuple=True)
        print(idx)
        features = self.bert(x)[0].squeeze()
        output = self.fc_model(features[idx, :])
        return output


class CheXPertFeaturizer(nn.Module):
    def __init__(self):
        super(CheXPertFeaturizer, self).__init__()
        model = models.densenet121(pretrained=True)
        model = drop_classification_layer(model)
        self.model = nn.Sequential(model, nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    CivilCommentsFeaturizer()