import os
import torch
import pandas as pd
import torch.utils.data as data

from PIL import Image
from torchvision import transforms
# from torchvision.datasets import CelebA

class AdultDataset(data.Dataset):
    def __init__(self, root, split="train"):
        datapath = os.path.join(root, "/adult")
        assert os.path.exists(datapath), "Adult dataset not found! Did you run `get_data.sh`?"

        self._filename = "adult.test" if split == "test" else "adult.data"
        data = pd.read_csv(datapath + "/" + self._filename, \
            names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',\
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',\
                   'hours-per-week', 'native-country', 'salary'])[1:]

        # One-hot encode categorical data
        for colum in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']:
            data = data[data[colum] != ' ?']
            onehot_colum = pd.get_dummies(data[colum], prefix=colum)
            data = pd.merge(left=data, right=onehot_colum, left_index=True, right_index=True)
            data = data.drop(columns=colum)
        
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        # Alias the datafram
        df = self._data
        df_x = df.loc[:, ~df.columns.isin(['salary_ <=50K', 'salary_ >50K', 'sex_ Female' ,'sex_ Male'])]
        x = torch.Tensor(df_x.values[i])
        t = torch.Tensor([df.iloc[i]['salary_ >50K']])
        d = torch.Tensor([df.iloc[i]['sex_ Male']])
        return x, t, d

class CheXpertDataset(data.Dataset):
    def __init__(self, root, split="train"):
        self._datapath = os.path.join(root, "chexpert")
        assert os.path.exists(self._datapath), "CheXpert dataset not found! Did you run `get_data.sh`?"
        
        self._filename = "train.csv" if split == "train" else "valid.csv"
        self._table = pd.read_csv(os.path.join(self._datapath, "CheXpert-v1.0-small", self._filename))

        self.transfrom = transforms.ToTensor()

    def __getitem__(self, i):
        # Alias the dataframe
        df = self._table

        # Get the image
        filename = df.iloc[i]['Path']
        img = Image.open(os.path.join(self._datapath, filename))

        x = self.transfrom(img)
        t = torch.Tensor([int(df.iloc[i]['No Finding'] != 1)]) # Count(1) = 22381, Count(nan) = 201033
        d = torch.Tensor([df.iloc[i]['Support Devices']]) # Count(1) = 116001, Count(nan) = 0,  Count(0.) = 6137, Count(-1.) = 1079
        return x, t, d

    def __len__(self):
        return len(self._table)

def get_train_validation_set(dataset:str, root="data/"):
    if dataset == "adult":
        train = AdultDataset(root, split="train")
        val = None
        return train, val
    else:
        pass

def get_test_set(dataset:str, root="data/"):
    if dataset == "adult":
        test = AdultDataset(root, split="test")
        return test
    else:
        pass

# if __name__ == "__main__":
#     dummy = CheXpertDataset("data")
#     a = dummy[1]
#     print(a)
#     print(len(dummy))
#     print(dummy[0]._data.head())
#     print(a)
#     # dummy2 = get_celeba()
