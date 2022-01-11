import os
import pandas as pd
import torch.utils.data as data
from torchvision.datasets import CelebA

class AdultDataset(data.Dataset):
    def __init__(self, root, split="train"):
        datapath = root + "/adult"
        assert os.path.exists(datapath), "Adult dataset not found! Did you run `get_data.sh`?"

        self._filename = "adult.test" if split == "test" else "adult.test"
        self._data = pd.read_csv(datapath + "/" + self._filename)

    def __getitem__(self, i):
        # For now the preprocessing steps are unknown, so we just return the table row
        item = self._data.values[i]
        return item


def get_adult(root="data"):
    train = AdultDataset(root, split="train")
    val = None
    test =  AdultDataset(root, split="test")
    return (train, val, test)

def get_celeba(root="data"):
    assert 5 == 0, "CelebA cannot be downloaded through pytorch. Please see https://github.com/pytorch/vision/issues/1920"
    train = CelebA(root=root, split="train", download=True)
    val = CelebA(root=root, split="val", download=True)
    test = CelebA(root=root, split="test", download=True)
    return (train, val, test)

    
def get_civil(root="data"):
    pass

def get_chexpert(root="data"):
    pass

if __name__ == "__main__":
    dummy = get_adult()
    dummy2 = get_celeba()