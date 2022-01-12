# FACT

## Datasets
The evalutation is done on four different datasets: [Adult](https://archive.ics.uci.edu/ml/datasets/adult), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Civil Comments](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) and [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/). This section describes how to obtain each of the datasets. The final data directory should look something like this

```
data
    / adult
        / adult.data
        / adult.train
    / celeba
        / ...
    / civil
        / ...
    / chexpert
        / ...
```

### Adult
The Adult dataset can be downloaded for free, without any registration using the [Adult dataset link](https://archive.ics.uci.edu/ml/datasets/adult), or by running the `data/get_adult.sh` (Linux and Mac only).

### CelebA
The [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is available for download through [Pytorch torchvision](https://pytorch.org/vision/stable/datasets.html#celeba), and thus does not need to be downloaded seperately.

### CheXpert
The [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) is available after registering. After recieving the download link, download the data to `data/chexpert`, or use `get_data.sh`.

### Civil Comments
The [Civil Comments](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) is available on Kaggle. To be able to download this dataset you
will need to register on Kaggle, verify on Kaggle with your telephone number and accept
the competition rules. Then you can download the API key in the profile section. This
file is called "kaggle.json" and this should be put in the ".kaggle" folder. Now you
can download the dataset using the ssh script. 
