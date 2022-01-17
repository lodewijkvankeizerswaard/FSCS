# FACT

## Downloading Datasets
The evalutation is done on four different datasets: [Adult](https://archive.ics.uci.edu/ml/datasets/adult), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Civil Comments](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) and [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/). This section describes how to obtain each of the datasets. If you just want to test the model in general, the Adult dataset is most easy to obtain and train on. The final data directory should look something like this:
```
data
    / adult
        / adult.data
        / adult.train
    / celeba
        / ...
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
The Adult dataset can be downloaded for free, without any registration using the [Adult dataset link](https://archive.ics.uci.edu/ml/datasets/adult), or by running the `data/get_adult.sh` (Linux and Mac only).

### CelebA
The [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is available for download through [Pytorch torchvision](https://pytorch.org/vision/stable/datasets.html#celeba), and thus does not need to be downloaded seperately.

### Civil Comments
The [Civil Comments](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) is available on Kaggle. To be able to download this dataset you will need to register on Kaggle, verify on Kaggle with your telephone number and accept the competition rules. Then you can download the API key in the profile section. This file is called "kaggle.json" and this should be put in the ".kaggle" folder. Now you can download the dataset using the ssh script (provided you use the environment specified in `environment.yml`). 

### CheXpert
The [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) is available after registering. After recieving the download link, download the small dataset to `data/chexpert` and unzip there, or insert the link into `get_data.sh` and run.

## Dataset preprocessing
The preprocessing for each of the datasets is done in `data.py`, in the individual `__init__` functions. This section will go over the exact preprocessing steps per dataset. Since we aim to replicate the work of [CITE], this part is not designed to be changed except for the attribute on which the sufficiency criterion is applied.

### Adult
For the Adult dataset we normalize the columns with continues values to have zero mean and unit variance. Furthermore, we one hot encode the columns with categorical values. The test data misses one categorical value for the "native-country" column, namely "Holand-Netherlands", which gets added and filled with all zeroes to be compatible with the training model. For some continues variables (i.e. 'fnlwgt', 'capital-gain' and 'capital-loss') a few outliers really skew the variance and mean and variance metrics, which should be further investigated.

### CheXpert
The images of the CheXpert data are cropped to 224 by 224 pixels (the bottom and right side information are thrown out), and this greyscale image is stacked three times to simulated R, G and B channels (since the DensNet121 expects this as input). The Pleural Effusion is the attribute for this dataset and has three possible values that correspond (in some way) to 'positive', 'negative' and 'undecided'. Everything that was not one, i.e. 'positive', was mapped to zero to make this a binary attribute.