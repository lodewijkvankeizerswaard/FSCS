import os
import torch
import pandas as pd
import torch.utils.data as data

from PIL import Image
from torchvision import transforms

# TODO add Civil Comments dataset object
# TODO add CelebA dataset object

class AdultDataset(data.Dataset):
    # TODO add docstrings
    # TODO add data bias 
    # TODO improve comments
    def __init__(self, root, split="train"):
        datapath = os.path.join(root, "adult")
        assert os.path.exists(datapath), "Adult dataset not found! Did you run `get_data.sh`?"

        self._filename = "adult.test" if split == "test" else "adult.data"
        table = pd.read_csv(datapath + "/" + self._filename, \
            names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',\
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',\
                   'hours-per-week', 'native-country', 'salary'])[1:]

        # One-hot encode categorical data
        for colum in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']:
            table = table[table[colum] != ' ?']
            onehot_colum = pd.get_dummies(table[colum], prefix=colum)
            table = pd.merge(left=table, right=onehot_colum, left_index=True, right_index=True)
            table = table.drop(columns=colum)

        # Normalize continous columns
        for column in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            m = table[column].mean()
            v = table[column].std()**2

            table[column] -= m
            table[column] /= v
        
        self._table = table

    def __len__(self):
        return len(self._table)

    def __getitem__(self, i):
        # Alias the datafram
        df = self._table
        df_x = df.loc[:, ~df.columns.isin(['salary_ <=50K', 'salary_ >50K', 'sex_ Female' ,'sex_ Male'])]
        x = torch.Tensor(df_x.values[i])
        t = torch.Tensor([df.iloc[i]['salary_ >50K']])
        d = torch.Tensor([df.iloc[i]['sex_ Male']])
        return x, t, d

class CheXpertDataset(data.Dataset):
    # TODO add docstring
    # TODO change target to Pleural Effusion
    # TODO image preprocessing
    # TODO improve comments
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

        x = self.transfrom(img)[:,:224,:224].repeat(3,1,1)
        t = torch.Tensor([int(df.iloc[i]['Pleural Effusion'] == 1)]) # Count(1) = 22381, Count(nan) = 201033
        d = torch.Tensor([int(df.iloc[i]['Support Devices'] == 1)]) # Count(1) = 116001, Count(nan) = 0,  Count(0.) = 6137, Count(-1.) = 1079
        return x, t, d

    def __len__(self):
        return len(self._table)

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
        pass

    return train, val

def get_test_set(dataset:str, root="data/"):
    # TODO add docstring
    # TODO add civil comments, chexpert, celeba
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

# def get_celeba(root = "data"):
#     data_root = root + "/celeba"
#     dataset_url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
#     download_path = f'{data_root}/img_align_celeba.zip'
#     dataset_folder = f'{root}/img_align_celeba'
#     if os.path.exists(download_path):
#         print("CelebA dataset already downloaded!")
#         return
#     if not os.path.exists(data_root):
#         os.makedirs(data_root)
#         os.makedirs(dataset_folder)
#     gdown.download(dataset_url, download_path, quiet = False)
#     with zipfile.ZipFile(download_path, 'r') as zip:
#         zip.extractall(dataset_folder)

# def preprocess_celeba(img_dir = "data\celeba\img_align_celeba\img_align_celeba"):
#     image_names = os.listdir(img_dir)
#     img_path = os.path.join(img_dir, image_names[0])
#     print(img_path)

# def validation_split(directory):
#     validation_data = pd.read_csv(directory, sep=" ", header=None)
    

class CelebADataset(data.Dataset):
    def __init__(self, root, split="train"):
        self._datapath = os.path.join(root, "celeba")
        assert os.path.exists(self._datapath), "CelebA dataset not found! Did you run 'get_data.sh'?"

        self.split_filename = "list_eval_partition.txt"
        self.anno_filename = "list_attr_celeba.txt"
        self.split_table = pd.read_csv(os.path.join(self._datapath, self.split_filename), sep="\t", header = 1)
        self.split_table.columns = ["image", "partition"]
        self.anno_table = pd.read_csv(os.path.join(self._datapath, self.anno_filename), sep="\t", header = 0)

        print(self.split_table.head())

        if split == "train":
            self.split_table = self.split_table[self.split_table.partition == 0]
        elif split == "valid":
            self.split_table = self.split_table[self.split_table.partition == 1]
        else:
            self.split_table = self.split_table[self.split_table.partition == 2]

        self.transform = transforms.ToTensor()
    
    def __getitem__(self, i):
        # Alias the dataframe and the evaluation partition
        df = self.anno_table
        df_split = self.split_table

        # Get the image from the training or test data
        filename = df_split.iloc[i]["image"]
        img = Image.open(os.path.join(self._datapath, "img_align_celeba", filename)) 

        resized_image = image.resize(224, 224)
        x = self.transfrom(img)
        t = torch.Tensor([int(df.iloc[i]['Blond_Hair'] == 1)])
        d = torch.Tensor([int(df.iloc[i]['Male'] == 1)])
        return x, t, d
     



        

    # class CheXpertDataset(data.Dataset):
    # # TODO add docstring
    # # TODO change target to Pleural Effusion
    # # TODO image preprocessing
    # # TODO improve comments
    # def __init__(self, root, split="train"):
    #     self._datapath = os.path.join(root, "chexpert")
    #     assert os.path.exists(self._datapath), "CheXpert dataset not found! Did you run `get_data.sh`?"
        
    #     self._filename = "train.csv" if split == "train" else "valid.csv"
    #     self._table = pd.read_csv(os.path.join(self._datapath, "CheXpert-v1.0-small", self._filename))

    #     self.transfrom = transforms.ToTensor()

    # def __getitem__(self, i):
    #     # Alias the dataframe
    #     df = self._table

    #     # Get the image
    #     filename = df.iloc[i]['Path']
    #     img = Image.open(os.path.join(self._datapath, filename))

    #     x = self.transfrom(img)[:,:224,:224].repeat(3,1,1)
    #     t = torch.Tensor([int(df.iloc[i]['Pleural Effusion'] == 1)]) # Count(1) = 22381, Count(nan) = 201033
    #     d = torch.Tensor([int(df.iloc[i]['Support Devices'] == 1)]) # Count(1) = 116001, Count(nan) = 0,  Count(0.) = 6137, Count(-1.) = 1079
    #     return x, t, d

    # def __len__(self):
    #     return len(self._table)




def get_civil(root="data"):
    pass

def get_chexpert(root="data"):
    pass

if __name__ == "__main__":
    print("hello")
    celeba = CelebADataset(root="data")
