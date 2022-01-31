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

from sklearn import preprocessing



class AdultDataset(data.Dataset):
    # TODO add docstrings
    # TODO improve comments
    def __init__(self, root, split="train"):
        features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"] 
        # self.feat_continous = ['Age', 'fnlwgt', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']
        # self.feat_categoric = ['Workclass', 'Education', 'Martial Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country', 'Target']

        # Change these to local file if available
        train_url = 'data/adult/adult.data'
        test_url = 'data/adult/adult.test'

        # This will download 3.8M
        original_train = pd.read_csv(train_url, names=features, sep=r'\s*,\s*', 
                                    engine='python', na_values="?")
        # This will download 1.9M
        original_test = pd.read_csv(test_url, names=features, sep=r'\s*,\s*', 
                                    engine='python', na_values="?", skiprows=1)

        
        
        original_train['Target'] = original_train['Target'].replace('<=50K', 0).replace('>50K', 1)
        original_test['Target'] = original_test['Target'].replace('<=50K.', 0).replace('>50K.', 1)
        original_train['Sex'] = original_train['Sex'].replace('Male', 1).replace('Female', 0)
        original_test['Sex'] = original_test['Sex'].replace('Male', 1).replace('Female', 0)

        # # Get rows where D=0 and Y=1, which we want to save
        bias_rows = original_train[(original_train["Sex"] == 1) & (original_train["Target"] == 0)]
        # Throw out all these rows 
        original_train = original_train[((original_train["Sex"] != 1) | (original_train["Target"] != 0))]
        # Add the 50 rows back
        original_train = pd.concat([original_train, bias_rows[:50]], ignore_index=True)

        # where_d_zero = set(original[ original["Sex"] == 1 ].index)
        # where_y_one = set(original[original["Target"] == 1].index)
        # drop_rows = list(where_d_zero & where_y_one)[50:]
        # print(num_train)
        # print(len(drop_rows))
        # original = original.drop(index=drop_rows)

        # shuffle the DataFrame rows
        # original = original.sample(frac = 1)

        num_train = len(original_train)

        original = pd.concat([original_train, original_test], ignore_index=True)

        labels = original["Sex"]
        attributes = original["Target"]

        del original['Target']

        # Find the ratio for the attribute to be able to sample from this distribution
        probs = self._attr_ratio(attributes)
        self._attr_dist = torch.distributions.categorical.Categorical(probs=probs)
        
        # Normalize continous columns
        table = self._data_transform(original)

        if split=="train":
            table = table[:num_train]
            attributes = attributes[:num_train]
            labels = labels[:num_train]
        else:
            table = table[num_train:]
            attributes = attributes[num_train:]
            labels = labels[num_train:]

        self._table = table
        self._attributes = attributes
        self._labels = labels

    def _data_transform(self, df):
        """Normalize features."""
        binary_data = pd.get_dummies(df)
        feature_cols = binary_data[binary_data.columns]
        # print(feature_cols)
        scaler = preprocessing.StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(feature_cols), columns=feature_cols.columns)
        return data

    def _attr_ratio(self, attributes: pd.DataFrame) -> torch.Tensor:
        """Finds the ratio in which the attribute occurs in the data set, such that we can later
        sample from this distribution. 
        Args:
            table (pd.DataFrame): the table from which to obtain the attribute ratio.
        Returns:
            torch.Tensor: a tensor with probabilities for the self.attribute['values'] in the same order.
        """
        ratios = attributes.value_counts(normalize=True).to_list()
        return torch.Tensor(ratios)

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
        return len(self._attributes.value_counts())

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
        return x, t, d

class CelebADataset(data.Dataset):
    def __init__(self, root, split="train"):
        self._datapath = os.path.join(root, "celeba")
        assert os.path.exists(self._datapath), "CelebA dataset not found! Did you run 'get_data.sh'?"

        self.split_filename = "list_eval_partition.txt"
        self.anno_filename = "list_attr_celeba.txt"
        self.split_table = pd.read_csv(os.path.join(self._datapath, self.split_filename), sep=" ", header = None, names=["image", "partition"])
        self.anno_table = pd.read_csv(os.path.join(self._datapath, self.anno_filename), sep=r"\s+", header = 1)

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

        self.transform = transforms.ToTensor()

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


        resized_image = img.resize((224, 224))

        x = self.transform(resized_image)
        t = torch.Tensor([int(df.iloc[i]['Blond_Hair'] == 1)])
        d = torch.Tensor([int(df.iloc[i]['Male'] == 1)])
        return x, t, d

class CivilDataset(data.Dataset):
    def __init__(self, root, split="train"):
        self._datapath = os.path.join(root, "civil")
        assert os.path.exists(self._datapath), "Civil dataset not found! Did you run 'get_data.sh'?"

        # Toxicity values were generated by a team of workers who assessed if a comment was toxic or not. Therefor there 
        # is not one binary value with toxic or not but a value between 0 and 1. Therefor we decided to set a threshold
        # at 0.5. All values below 0.5 were classified as 0 (not toxic) and all values above 0.5 were classified as 1
        self._filename = "train.csv" if split == "train" else "test.csv"
        self._alldata_filename = "all_data.csv"
        self._partition_table = pd.read_csv(os.path.join(self._datapath, self._filename))
        self._alldata_table = pd.read_csv(os.path.join(self._datapath, self._alldata_filename))

        self.attribute = {'column' : 'christian', 'values' : [0, 1]}
        index_list = list(self._partition_table.index.values)
        self._alldata_table = self._alldata_table.iloc[index_list]
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
    print("Running all dataset objects!")
    train_set = AdultDataset('data', split="train")
    # val_set = AdultDataset('data', split="val")
    # test_set = AdultDataset('data', split="test")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                               shuffle=True, num_workers=3, drop_last=True)
    # print(len(train_set), len(val_set), len(test_set))
    # train_set.sample_d((10,10))
    # for i, p in enumerate(tqdm(train_loader)):
    #     a = p[1]
    #     print(a)
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