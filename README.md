# FSCS

This is the repository for our entry in the (ML Reproducibility Challenge 2021 Fall Edition)[https://paperswithcode.com/rc2021], where we reproduce the paper "Fair Selective Classification via Sufficiency". 

## Downloading Datasets
The evalutation is done on four different datasets: [Adult](https://archive.ics.uci.edu/ml/datasets/adult), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Civil Comments](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) and [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/). This section describes how to obtain each of the datasets. If you just want to test the model in general, the Adult dataset is most easy to obtain and train on. The final data directory should look something like this:
```
data
    / adult
        / adult.data
        / adult.train
    / celeba
        / img_align_celeba
        / list_attr_celeba.txt
        / list_eval_partition.txt
    / chexpert
        / CheXpert-v1.0small
            / train
            / valid
            / train.csv
            / valid.csv
    / civil
        / all_data.csv
        / identity_individual_annotations.csv
        / sample_submission.csv
        / test.csv
        / test_private_expanded.csv
        / test_public_expanded.csv
        / toxicity_individual_annotations.csv
        / train.csv

```

### Adult
The Adult dataset can be downloaded for free, without any registration using the [Adult dataset link](https://archive.ics.uci.edu/ml/datasets/adult). The dataset we used was preprocessed using [this repo](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_optim_preproc_adult.ipynb), and is available in the data directory.

### CelebA
The [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is available for download through [Google Drive via their website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

### Civil Comments
The [Civil Comments](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) is available on Kaggle. To be able to download this dataset you will need to register on Kaggle, verify on Kaggle with your telephone number and accept the competition rules. Then you can download the API key in the profile section. This file is called "kaggle.json" and this should be put in the ".kaggle" folder. 

### CheXpert
The [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) is available after registering. After recieving the download link, download the small dataset to `data/chexpert` and unzip there.

## Dataset preprocessing
The preprocessing for each of the datasets is done in `data.py`, in the individual `__init__` functions. This section will go over the exact preprocessing steps per dataset. Since we aim to replicate the work of [CITE], this part is not designed to be changed except for the attribute on which the sufficiency criterion is applied.

### Adult
For the Adult dataset we normalize the columns with continues values to have zero mean and unit variance. Furthermore, we one hot encode the columns with categorical values. The test data misses one categorical value for the "native-country" column, namely "Holand-Netherlands", which gets added and filled with all zeroes to be compatible with the training model. For some continues variables (i.e. 'fnlwgt', 'capital-gain' and 'capital-loss') a few outliers really skew the variance and mean and variance metrics, which should be further investigated.

### CheXpert
The images of the CheXpert data are cropped to 224 by 224 pixels (the bottom and right side information are thrown out), and this greyscale image is stacked three times to simulated R, G and B channels (since the DensNet121 expects this as input). The Pleural Effusion is the attribute for this dataset and has three possible values that correspond (in some way) to 'positive', 'negative' and 'undecided'. Everything that was not one, i.e. 'positive', was mapped to zero to make this a binary attribute.

### CelebA
The images of the Celeba dataset are resized to 224 by 224 pixels by using scaling. Both the attributes and the dataset partition were taken from text files. The attribute text file contained an non uniform amount of whitespaces which was important to keep in mind when trying to use pandas. 

### Civil Comments
The Civil Comments dataset contained a lot of rows where the attribute "Christian" was not defined. Because this was the sensitive attribute we had to use all rows where this was the case were removed. The toxicity value was a continous value between 0 and 1. Therefor a threshold was set at 0.5, all values lower than this were seen as non-toxic and all values higher were seen as toxic. 


## Running the code
The `train_model.py` can be used to train a model with the specified parameters e.g. `python train_model.py --dataset adult --lmbda 0.7 --optimizer adam --seed 42 --progress_bar`. Using the default settings will train a regularized model for the specified dataset for 20 epochs (this should be adjusted for the specific dataset). Please see our paper for more details on hyperparameter tuning.

A trained model can be copied to the models directory, and then evaluated using the evaluate function in `results.ipynb`.
