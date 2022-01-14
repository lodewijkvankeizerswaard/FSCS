import os
import torch
import pandas as pd
import torch.utils.data as data
import zipfile
import gdown
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
        t = torch.Tensor([df.iloc[i]['salary_ >50K']])
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

# class CelebA(data.Dataset):
#     def __init__(self, root, split = "train"):
#         datapath = root + "/celeba"
#         assert os.path.exists(datapath), "CelebA dataset not found! Did you run 'get_data.sh'?"
#
#         self._filename = "celeba.test" if split == "test" else "adult.data"
#         self.url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'


# def get_celeba(root="data"):
#     assert 5 == 0, "CelebA cannot be downloaded through pytorch. Please see https://github.com/pytorch/vision/issues/1920"
#     train = CelebA(root=root, split="train", download=True)
#     val = CelebA(root=root, split="val", download=True)
#     test = CelebA(root=root, split="test", download=True)
#     return (train, val, test)

def get_celeba(root = "data"):
    data_root = root + "/celeba"
    dataset_url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
    download_path = f'{data_root}/img_align_celeba.zip'
    dataset_folder = f'{root}/img_align_celeba'
    if os.path.exists(download_path):
        print("CelebA dataset already downloaded!")
        return
    if not os.path.exists(data_root):
        os.makedirs(data_root)
        os.makedirs(dataset_folder)
    gdown.download(dataset_url, download_path, quiet = False)
    with zipfile.ZipFile(download_path, 'r') as zip:
        zip.extractall(dataset_folder)


def get_civil(root="data"):
    pass

def get_chexpert(root="data"):
    pass

if __name__ == "__main__":
    dummy = get_celeba()
