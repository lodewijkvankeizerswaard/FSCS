import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import zipfile
import gdown
from PIL import Image
from torchvision import transforms


# TODO add Civil Comments dataset object
# TODO add CelebA dataset object

# Dataset attribute selection. Please make sure that the column name and column values are correct.
CHEXPERT_ATTRIBUTE = {'column' : 'Pleural Effusion', 'values' : [0, 1]}
ADULT_ATTRIBUTE = {'column' : 'sex', 'values' : [' Male', ' Female']}
CELEBA_ATTRIBUTE = {'column' : 'Male', 'values' : [-1, 1]}
# ADULT_ATTRIBUTE = {'column' : 'relationship', 'values' : [' Husband', ' Not-in-family', ' Wife', ' Own-child', ' Unmarried', ' Other-relative']}

# Editing these global variables has a very high chance of breaking the data
ADULT_CATEGORICAL = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']
ADULT_CONTINOUS = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

class AdultDataset(data.Dataset):
    # TODO add docstrings
    # TODO add data bias 
    # TODO improve comments
    def __init__(self, root, split="train"):
        datapath = os.path.join(root, "adult")
        assert os.path.exists(datapath), "Adult dataset not found! Did you run `get_data.sh`?"

        # Read data and skip first line of test data
        self._filename = "adult.test" if split == "test" else "adult.data"
        table = pd.read_csv(os.path.join(datapath, self._filename), \
            names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',\
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',\
                   'hours-per-week', 'native-country', 'salary'], skiprows=int(split=="test"))

        # Remove dots from labels (in test data)
        table['salary'] = table['salary'].str.replace('.', '', regex=False)

        # One-hot encode categorical data
        table = self._onehot_cat(table, ADULT_CATEGORICAL)

        # Add missing country to test data
        if split == "test":
            table['native-country_ Holand-Netherlands'] = np.zeros(len(table))

        # Normalize continous columns
        table = self._normalize_con(table, ADULT_CONTINOUS)

        # Find the ratio for the attribute to be able to sample from this distribution
        probs = self._attr_ratio(table)
        self._attr_dist = torch.distributions.categorical.Categorical(probs=probs)

        self._table = table

    def _attr_ratio(self, table: pd.DataFrame) -> torch.Tensor:
        """Finds the ratio in which the attribute occurs in the data set, such that we can later
        sample from this distribution. 

        Args:
            table (pd.DataFrame): the table from which to obtain the attribute ratio.

        Returns:
            torch.Tensor: a tensor with probabilities for the ADULT_ATTRIBUTE['values'] in the same order.
        """
        counts = table[ADULT_ATTRIBUTE['column']].value_counts()
        ratios = torch.Tensor([counts[attr_val] for attr_val in ADULT_ATTRIBUTE['values']])
        return ratios / sum(ratios)

    def sample_d(self, size: tuple) -> torch.Tensor:
        return self._attr_dist.sample(size)

    def _onehot_cat(self, table: pd.DataFrame, categories: list) -> pd.DataFrame:
        """One hot encodes the columns of the table for which the names are in categories

        Args:
            table (pd.DataFrame): the table containing the data
            categories (list): a list of column names which to encode

        Returns:
            pd.DataFrame: the table object with the one hot encoded columns appended, and the original column removed
        """
        for column in categories:
            table = table[table[column] != ' ?']
            onehot_colum = pd.get_dummies(table[column], prefix=column)
            table = pd.merge(left=table, right=onehot_colum, left_index=True, right_index=True)
            if column != ADULT_ATTRIBUTE['column']:
                table = table.drop(columns=column)
        return table

    def _normalize_con(self, table: pd.DataFrame, categories: list) -> pd.DataFrame:
        """Normalizes the columns of a table to have zero mean and unit variance.

        Args:
            table (pd.Dataframe): the table containing the data.
            categories (list): a list of column names present in the table that need to be normalized.

        Returns:
            pd.Dataframe: the table with the given columns normalized.
        """
        for column in categories:
            m = table[column].mean()
            v = table[column].std()**2

            table[column] -= m
            table[column] /= v
        return table

    def datapoint_shape(self) -> torch.Tensor:
        """Return the amount of elements in each x value

        Returns:
            int: the amount of elements in x
        """
        return self[0][0].shape

    def nr_attr_values(self) -> int:
        """Returns the number of possible values for the attribute of this dataset.

        Returns:
            int: the number of attributes
        """
        return len(ADULT_ATTRIBUTE['values'])

    def __len__(self) -> int:
        """Returns the amount of datapoints in this data object."""
        return len(self._table)

    def __getitem__(self, i: int) -> tuple:
        """Gets the i-th element from the table. 

        Args:
            i (int): item number

        Returns:
            tuple: The x value includes all one hot encoded and continous data except for the target 
        value, and the column that contains the non-one hot encoded attribute (since this is only used as a map for d). The t 
        value is binary (whether this person earns more than 50K). The d value is a value that indicates the element number in
        the ADULT_ATTRIBUTE['values'] list. This determines the mapping for the group specific model later on.
        """
        # Alias the datafram
        df = self._table
        # x is all values, except the target value, and the attribute column
        df_x = df.loc[:, ~df.columns.isin(['salary_ <=50K', 'salary_ >50K', ADULT_ATTRIBUTE['column']])]
        x = df_x.values[i]
        t = [df.iloc[i]['salary_ >50K']]
        d = [ ADULT_ATTRIBUTE['values'].index(df.iloc[i][ ADULT_ATTRIBUTE['column'] ])]
        return torch.Tensor(x), torch.Tensor(t), torch.Tensor(d)

class CheXpertDataset(data.Dataset):
    # TODO add docstring
    # TODO change target to Pleural Effusion
    # TODO image preprocessing
    # TODO improve comments
    def __init__(self, root, split="train"):
        self._datapath = os.path.join(root, "chexpert")
        assert os.path.exists(self._datapath), "CheXpert dataset not found! Did you run `get_data.sh`?"
        
        # Read the csv file, and replace all 0's and -1's with nan (to make flags binary)
        self._filename = "train.csv" if split == "train" else "valid.csv"
        self._table = pd.read_csv(os.path.join(self._datapath, "CheXpert-v1.0-small", self._filename), na_values=[0,-1])
        self._table.fillna(0, inplace=True)

        # Find the ratio for the attribute to be able to sample from this distribution
        probs = self._attr_ratio(self._table)
        self._attr_dist = torch.distributions.Categorical(probs=probs)

        self._transfrom = transforms.ToTensor()

    def _attr_ratio(self, table: pd.DataFrame) -> torch.Tensor:
        """Finds the ratio in which the attribute occurs in the data set, such that we can later
        sample from this distribution. 

        Args:
            table (pd.DataFrame): the table from which to obtain the attribute ratio.

        Returns:
            torch.Tensor: a tensor with probabilities for the ADULT_ATTRIBUTE['values'] in the same order.
        """
        counts = table[CHEXPERT_ATTRIBUTE['column']].value_counts()
        ratios = torch.Tensor([counts[attr_val] for attr_val in CHEXPERT_ATTRIBUTE['values']])
        return ratios / sum(ratios)

    def sample_d(self, size: tuple) -> torch.Tensor:
        return self._attr_dist.sample(size)

    def datapoint_shape(self) -> torch.Tensor:
        """Return the amount of elements in each x value

        Returns:
            int: the amount of elements in x
        """
        return self[0][0].shape

    def nr_attr_values(self) -> int:
        """Returns the number of possible values for the attribute of this dataset.

        Returns:
            int: the number of attributes
        """
        return len(CHEXPERT_ATTRIBUTE['values'])
    
    def __len__(self):
        return len(self._table)

    def __getitem__(self, i):
        # Alias the dataframe
        df = self._table

        # Get the image
        filename = df.iloc[i]['Path']
        img = Image.open(os.path.join(self._datapath, filename))

        x = self._transfrom(img)[:,:224,:224].repeat(3,1,1)
        t = torch.Tensor([int(df.iloc[i]['Pleural Effusion'] == 1)]) # Count(1) = 22381, Count(nan) = 201033
        d = torch.Tensor([int(df.iloc[i]['Support Devices'] == 1)]) # Count(1) = 116001, Count(nan) = 0,  Count(0.) = 6137, Count(-1.) = 1079
        return x, t, d

def get_train_validation_set(dataset:str, root="data/"):
    # TODO add docstring
    # TODO add civil comments, chexpert, celeba
    if dataset == "adult":
        train = AdultDataset(root, split="train")
        val = None
    elif dataset == "chexpert":
        train = CheXpertDataset(root, split="train")
        val = None
    else:
        raise ValueError("This dataset is not implemented") 
    return train, val

def get_test_set(dataset:str, root="data/"):
    # TODO add docstring
    # TODO add civil comments, chexpert, celeba
    if dataset == "adult":
        test = AdultDataset(root, split="test")
    elif dataset == "chexpert":
        test = CheXpertDataset(root, split="test")
    else:
        raise ValueError("This dataset is not implemented")
    return test

    

class CelebADataset(data.Dataset):
    def __init__(self, root, split="train"):
        self._datapath = os.path.join(root, "celeba")
        assert os.path.exists(self._datapath), "CelebA dataset not found! Did you run 'get_data.sh'?"

        self.split_filename = "list_eval_partition.txt"
        self.anno_filename = "list_attr_celeba.txt"
        self.split_table = pd.read_csv(os.path.join(self._datapath, self.split_filename), sep=" ", header = None, names=["image", "partition"])
        self.anno_table = pd.read_csv(os.path.join(self._datapath, self.anno_filename), sep=r"\s+", header = 1)

        if split == "train":
            self.split_table = self.split_table[self.split_table.partition == 0]
        elif split == "valid":
            self.split_table = self.split_table[self.split_table.partition == 1]
        else:
            self.split_table = self.split_table[self.split_table.partition == 2]

        index_list = list(self.split_table.index.values)
        self.anno_table = self.anno_table.iloc[index_list]

        probs = self._attr_ratio(self.anno_table)
        self._attr_dist = torch.distributions.Categorical(probs=probs)

        self.transform = transforms.ToTensor()

    def _attr_ratio(self, table: pd.DataFrame) -> torch.Tensor:
        """Finds the ratio in which the attribute occurs in the data set, such that we can later
        sample from this distribution. 

        Args:
            table (pd.DataFrame): the table from which to obtain the attribute ratio.

        Returns:
            torch.Tensor: a tensor with probabilities for the ADULT_ATTRIBUTE['values'] in the same order.
        """
        counts = table[CELEBA_ATTRIBUTE['column']].value_counts()
        ratios = torch.Tensor([counts[attr_val] for attr_val in CELEBA_ATTRIBUTE['values']])
        return ratios / sum(ratios) 

    def sample_d(self, size: tuple) -> torch.Tensor:
        return self._attr_dist.sample(size)

    def datapoint_shape(self) -> torch.Tensor:
        """Return the amount of elements in each x value

        Returns:
            int: the amount of elements in x
        """
        return self[0][0].shape

    def nr_attr_values(self) -> int:
        """Returns the number of possible values for the attribute of this dataset.

        Returns:
            int: the number of attributes
        """
        return len(CELEBA_ATTRIBUTE['values'])

    def __len__(self):
        return len(self._table)

    def __getitem__(self, i):
        # Alias the dataframe and the evaluation partition
        df = self.anno_table
        df_split = self.split_table


        # Get the image from the training or test data
        filename = df_split.iloc[i]["image"]
        img = Image.open(os.path.join(self._datapath, "img_align_celeba", filename)) 


        resized_image = img.resize((224, 224))

        x = self.transform(resized_image)
        t = torch.Tensor([int(df.iloc[i]['Blond_Hair'] == 1)])
        d = torch.Tensor([int(df.iloc[i]['Male'] == 1)])
        return x, t, d




def get_civil(root="data"):
    pass

def get_chexpert(root="data"):
    pass

if __name__ == "__main__":
    celeba = CelebADataset(root="data")
    print(celeba.datapoint_shape)
