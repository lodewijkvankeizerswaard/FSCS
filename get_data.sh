#!/bin/bash

# This script downloads all datasets used for this project into the `data` 
# directory. To be able to obtain all data, some preparing steps have to be 
# taken. Please see the dataset specific steps in `README.md`.

cd data

#################
# ADULT DATASET #
#################
if [[ -f adult/adult.data && -f adult/adult.test ]]; then
    echo "Adult dataset found!"
else
    # Make the directory
    [[ ! -d "adult" ]] &&  mkdir adult
    # Download the files
    cd adult
    [[ ! -f "adult.data" ]] && wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    [[ ! -f "adult.test" ]] && wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
    cd ..
fi

####################
# CHEXPERT DATASET #
####################
URL="" # See README.md
if [[ -f chexpert/train.csv && -f chexpert/valid.csv && -d chexpert/train && chexpert/valid ]]; then
    echo "CheXpert dataset found!"
else
    # Make the directory
    [[ ! -d "chexpert" ]] &&  mkdir chexpert
    # Download the files
    cd chexpert
    [[ ! -f "CheXpert-v1.0-small.zip" ]] && wget $URL
    unzip -u CheXpert-v1.0-small.zip
    rm CheXpert-v1.0-small.zip
    cd ..
fi

# CIVIL COMMENTS DATASET
# Before starting you need to install kaggle with pip or anaconda
# First you need to register and verify on Kaggle and accept the competition rules.
# Then you need to download the API token and put it in the .kaggle folder of your user account.
# This file is called kaggle.json
[[ ! -d "civil" ]] &&  mkdir civil
cd civil
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
cd ..

##################
# CELEBA DATASET #
##################
CELEB_URL="'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'"
CELEB_ANNO_URL = "'https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U'"
CELEB_EVAL_URL = "'https://drive.google.com/uc?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk"
if [[ -d celeba/img_align_celeba ]] && [[-d celeba/anno]]; then
    echo "CelebA dataset found!"
else
    [[ ! -d "celeba" ]] &&  mkdir celeba
    cd celeba
    [[ ! -f "img_align_celeba.zip" ]] && python -c "import gdown; gdown.download($CELEB_URL, '.')"
    unzip -u img_align_celeba.zip
    rm img_align_celeba.zip
    cd .. 
    [[ ! -d "anno"]] && mkdir anno
    cd anno 
    [[ ! -f "list_atr_celeba.txt" ]] && python -c "import gdown; gdown.download($CELEB_ANNO_URL, '.')"
    [[ ! -f "list_eval_partition.txt" ]] && python -c "import gdown; gdown.download($CELEB_EVAL_URL, '.')"
fi
