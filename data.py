import os
import torch
import pandas as pd
import torch.utils.data as data
# from torchvision.datasets import CelebA

class AdultDataset(data.Dataset):
    def __init__(self, root, split="train"):
        datapath = root + "/adult"
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
        return len(self._data.values)

    def __getitem__(self, i):
        # Alias the datafram
        df = self._data
        df_x = df.loc[:, ~df.columns.isin(['salary_ <=50K', 'salary_ >50K', 'sex_ Female' ,'sex_ Male'])]
        x = torch.Tensor(df_x.values[i])
        t = torch.Tensor([df.iloc[i]['salary_ <=50K'], df.iloc[i]['salary_ >50K']])
        d = torch.Tensor([df.iloc[i]['sex_ Male']])
        return x, t, d


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

# def get_celeba(root="data"):
#     assert 5 == 0, "CelebA cannot be downloaded through pytorch. Please see https://github.com/pytorch/vision/issues/1920"
#     train = CelebA(root=root, split="train", download=True)
#     val = CelebA(root=root, split="val", download=True)
#     test = CelebA(root=root, split="test", download=True)
#     return (train, val, test)

    
def get_civil(root="data"):
    pass

def get_chexpert(root="data"):
    pass

# if __name__ == "__main__":
#     dummy = get_adult()
#     a = dummy[0][1]
#     print(dummy[0]._data.head())
#     print(a)
#     # dummy2 = get_celeba()
