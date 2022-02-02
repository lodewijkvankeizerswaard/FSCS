from audioop import bias
import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import zipfile
# import gdown
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# TODO add Civil Comments dataset object
# TODO add CelebA dataset object

# Editing these global variables has a very high chance of breaking the data
ADULT_CATEGORICAL = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']
ADULT_CONTINOUS = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

class AdultDataset(data.Dataset):
    # TODO add docstrings
    # TODO improve comments
    def __init__(self, root='data', split="train", attribute='sex'):
        datapath = os.path.join(root, "adult")
        assert os.path.exists(datapath), "Adult dataset not found! Did you run `get_data.sh`?"

        # Read data and skip first line of test data
        self._filename = "adult.test" if split == "test" else "adult.data"
        table = pd.read_csv(os.path.join(datapath, self._filename), skipinitialspace=True, \
            names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',\
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',\
                   'hours-per-week', 'native-country', 'salary'],  skiprows=int(split=="test"))
        
        self.attribute = attribute

        # Remove dots from labels (in test data)
        table['salary'] = table['salary'].str.replace('.', '', regex=False)

        # One-hot encode categorical data
        table = self._onehot_cat(table, ADULT_CATEGORICAL)

        if split == "test":
            # Add missing country to test data
            table['native-country_Holand-Netherlands'] = np.zeros(len(table))
        else:
            # # Introduce bias in the train data
            drop_rows = table[(table['salary_>50K'] == 1) & (table['sex_Male'] == 0)].index[50:]
            table = table.drop(index=drop_rows)
            
        
        # Normalize continous columns
        table = self._normalize_con(table, ADULT_CONTINOUS)

        self._attributes = table['sex_Male']
        # # remove attribute from table
        # del table['sex_Female']
        # del table['sex_Male']

        self._labels = table["salary_>50K"]
        del table['salary_<=50K']
        del table['salary_>50K']

        self._table = table

        # Find the ratio for the attribute to be able to sample from this distribution
        probs = self._attr_ratio(table)
        self._attr_dist = torch.distributions.categorical.Categorical(probs=probs)

    def _attr_ratio(self, table: pd.DataFrame) -> torch.Tensor:
        """Finds the ratio in which the attribute occurs in the data set, such that we can later
        sample from this distribution. 

        Args:
            table (pd.DataFrame): the table from which to obtain the attribute ratio.

        Returns:
            torch.Tensor: a tensor with probabilities for the self.attribute['values'] in the same order.
        """
        counts = self._attributes.value_counts()
        return torch.Tensor(counts / sum(counts))

    def sample_d(self, size: tuple) -> torch.Tensor:
        return self._attr_dist.sample(size).squeeze()

    def _onehot_cat(self, table: pd.DataFrame, categories: list) -> pd.DataFrame:
        """One hot encodes the columns of the table for which the names are in categories

        Args:
            table (pd.DataFrame): the table containing the data
            categories (list): a list of column names which to encode

        Returns:
            pd.DataFrame: the table object with the one hot encoded columns appended, and the original column removed
        """
        for column in categories:
            table = table[table[column] != '?']
            onehot_colum = pd.get_dummies(table[column], prefix=column)
            table = pd.merge(left=table, right=onehot_colum, left_index=True, right_index=True)
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
            min_val = table[column].min()
            max_val = table[column].max()

            table[column] = (table[column] - min_val) / (max_val - min_val)
            # table[column] -= table[column].mean()
            # table[column] /= table[column].var()
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
        return len(self._attributes.unique())

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
        the self.attribute['values'] list. This determines the mapping for the group specific model later on.
        """
        x = self._table.iloc[i]
        t = self._labels.iloc[i]
        d = self._attributes.iloc[i]
        return torch.Tensor(x), torch.Tensor([t]).squeeze(), torch.Tensor([d])

class CheXpertDataset(data.Dataset):
    # TODO add docstring
    # TODO improve comments
    def __init__(self, root, split="train"):
        self._datapath = os.path.join(root, "chexpert")
        assert os.path.exists(self._datapath), "CheXpert dataset not found! Did you run `get_data.sh`?"
        self.attribute = {'column' : 'Support Devices', 'values' : [0, 1]}
        self.target = {'column' : "Pleural Effusion", 'values' : [0 ,1]}
        
        # Read the csv file, and 
        self._filename = "train.csv" if split == "train" else "valid.csv"
        self._table = pd.read_csv(os.path.join(self._datapath, "CheXpert-v1.0-small", self._filename))

        # Remove rows with -1's for the attribute value and target value (to make flags binary)
        self._table = self._table[ self._table[self.attribute['column']].isin(self.attribute['values']) == True ]
        self._table = self._table[ self._table[self.target['column']].isin(self.target['values']) == True ]
        print(self._table['Pleural Effusion'].unique())
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
            torch.Tensor: a tensor with probabilities for the self.attribute['values'] in the same order.
        """
        counts = table[self.attribute['column']].value_counts()
        ratios = torch.Tensor([counts[attr_val] for attr_val in self.attribute['values']])
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
        return len(self.attribute['values'])
    
    def __len__(self):
        return len(self._table)

    def __getitem__(self, i):
        # Alias the dataframe
        df = self._table

        # Get the image
        filename = df.iloc[i]['Path']
        img = Image.open(os.path.join(self._datapath, filename)).resize((224,224))

        x = self._transfrom(img).repeat(3,1,1)
        t = torch.Tensor([int(df.iloc[i]['Pleural Effusion'] == 1)])
        d = torch.Tensor([int(df.iloc[i][self.attribute['column']] == 1)])
        return x, t.squeeze(), d.squeeze()

class CelebADataset(data.Dataset):
    def __init__(self, root, split="train"):
        self._datapath = os.path.join(root, "celeba")
        assert os.path.exists(self._datapath), "CelebA dataset not found! Did you run 'get_data.sh'?"

        self.split_filename = "list_eval_partition.txt"
        self.anno_filename = "list_attr_celeba.txt"
        self.split_table = pd.read_csv(os.path.join(self._datapath, self.split_filename), sep=" ", header = None, names=["image", "partition"])
        self.anno_table = pd.read_csv(os.path.join(self._datapath, self.anno_filename), sep=r"\s+", header = 1)

        # Split the dataset into a train, validation and test dataset.
        self.attribute = {'column' : 'Male', 'values' : [-1, 1]}

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

        self.transform = transforms.Compose([
                               transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def _attr_ratio(self, table: pd.DataFrame) -> torch.Tensor:
        """Finds the ratio in which the attribute occurs in the data set, such that we can later
        sample from this distribution. 

        Args:
            table (pd.DataFrame): the table from which to obtain the attribute ratio.

        Returns:
            torch.Tensor: a tensor with probabilities for the self.attribute['values'] in the same order.
        """
        counts = table[self.attribute['column']].value_counts()
        ratios = torch.Tensor([counts[attr_val] for attr_val in self.attribute['values']])
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
        return len(self.attribute['values'])

    def __len__(self):
        return len(self.anno_table)

    def __getitem__(self, i):
        # Alias the dataframe and the evaluation partition
        df = self.anno_table
        df_split = self.split_table


        # Get the image from the training or test data
        filename = df_split.iloc[i]["image"]
        img = Image.open(os.path.join(self._datapath, "img_align_celeba", filename)) 

        # Resize the image to the dimension required by the Resnet50

        x = self.transform(img)
        t = torch.Tensor([int(df.iloc[i]['Blond_Hair'] == 1)])
        d = torch.Tensor([int(df.iloc[i]['Male'] == 1)])
        return x, t.squeeze(), d.squeeze()

class CivilDataset(data.Dataset):
    def __init__(self, root, split="train"):
        self._datapath = os.path.join(root, "civil")
        assert os.path.exists(self._datapath), "Civil dataset not found! Did you run 'get_data.sh'?"

        self._filename = "train.csv" if split == "train" else "test.csv"
        self._alldata_filename = "all_data.csv"
        self._partition_table = pd.read_csv(os.path.join(self._datapath, self._filename))
        self._alldata_table = pd.read_csv(os.path.join(self._datapath, self._alldata_filename))

        self.attribute = {'column' : 'christian', 'values' : [0, 1]}
        index_list = list(self._partition_table.index.values)
        self._alldata_table = self._alldata_table.iloc[index_list]
        # Remove all columns where the attribute Christian is not defined
        self._alldata_table = self._alldata_table[self._alldata_table['christian'].notna()]
        self._alldata_table.sort_values(by="comment_text", key=lambda x: x.str.len())


        probs = self._attr_ratio(self._alldata_table)
        self._attr_dist = torch.distributions.Categorical(probs = probs)

        self._transform = transforms.ToTensor()

    def _attr_ratio(self, table: pd.DataFrame) -> torch.Tensor:
        """Finds the ratio in which the attribute occurs in the data set, such that we can later
        sample from this distribution. 

        Args:
            table (pd.DataFrame): the table from which to obtain the attribute ratio.

        Returns:
            torch.Tensor: a tensor with probabilities for the self.attribute['values'] in the same order.
        """
        counts = table[self.attribute['column']].value_counts()
        ratios = torch.Tensor([counts[attr_val] for attr_val in self.attribute['values']])
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
        return len(self.attribute['values'])

    def __len__(self):
        return len(self._alldata_table)
    
    def __getitem__(self, i):
        #Alias the dataframe
        df = self._alldata_table

        # Toxicity values were generated by a team of workers who assessed if a comment was toxic or not. Therefor there 
        # is not one binary value with toxic or not but a continous value between 0 and 1. Therefor we decided to set a threshold
        # at 0.5. All values below 0.5 were classified as 0 (not toxic) and all values above 0.5 were classified as 1
        x = df.iloc[i]['comment_text']
        t = int(df.iloc[i]['toxicity'] >= 0.5)
        d = int(df.iloc[i]['christian'] == 1)
        # x = self.tokenizer.encode(x, padding='max_length', max_length=512, return_tensors='pt')
        return x, torch.Tensor([t]), torch.Tensor([d])

def get_train_validation_set(dataset:str, root="data/", attribute=""):
    # TODO add docstring
    # TODO add attribute passthrough to dataset objects
    if dataset == "adult":
        train = AdultDataset(root, split="train")
        val = None
    elif dataset == "chexpert":
        train = CheXpertDataset(root, split="train")
        val = None
    elif dataset == "celeba":
        train = CelebADataset(root, split="train")
        val = CelebADataset(root, split = "valid")
    elif dataset == "civil":
        train = CivilDataset(root, split="train")
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
    elif dataset == "celeba":
        test = CelebADataset(root, split="test")
    elif dataset == "civil":
        test = CivilDataset(root, split="test")
    else:
        raise ValueError("This dataset is not implemented")
    return test

if __name__ == "__main__":
    # print("Running all dataset objects!")
    # train_set = AdultDataset('data', split="train")
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
    #                                            shuffle=True, num_workers=3, drop_last=True)
    # train_set.sample_d((10,10))
    # for i, p in enumerate(tqdm(train_loader)):
    #     a = p[0]
    #     if i > 10:
    #         break

    # train_set = CelebADataset('data', split="train")
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
    #                                            shuffle=True, num_workers=3, drop_last=True)
    # train_set.sample_d((10,10))
    # for i, p in enumerate(tqdm(train_loader)):
    #     a = p[0]
    #     if i > 10:
    #         break

    # train_set = CheXpertDataset('data', split="train")
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
    #                                            shuffle=True, num_workers=3, drop_last=True)
    # train_set.sample_d((10,10))
    # for i, p in enumerate(tqdm(train_loader)):
    #     a = p[0]
    #     if i > 10:
    #         break

    # train_set = CivilDataset('data', split="train")
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
    #                                            shuffle=True, num_workers=3, drop_last=True)
    # train_set.sample_d((10,10))
    # for i, p in enumerate(tqdm(train_loader)):
    #     a = p[0]
    #     if i > 10:
    #         break
    AdultDataset()