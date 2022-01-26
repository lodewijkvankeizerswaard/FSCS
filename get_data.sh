#!/bin/bash

# This script downloads all datasets used for this project into the `data`
# directory. To be able to obtain all data, some preparing steps have to be
# taken. Please see the dataset specific steps in `README.md`.

cd data
source activate FSCS

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
CHXPRT_URL="" # See README.md
CHXPRT_DIR="chexpert/CheXpert-v1.0-small"
if [[ -f $CHXPRT_DIR/train.csv && -f $CHXPRT_DIR/valid.csv && -d $CHXPRT_DIR/train && $CHXPRT_DIR/valid ]]; then
    echo "CheXpert dataset found!"
else
    # Make the directory
    [[ ! -d "chexpert" ]] &&  mkdir chexpert
    # Download the files
    cd chexpert
    [[ ! -f "CheXpert-v1.0-small.zip" ]] && wget $CHXPRT_URL
    unzip -u CheXpert-v1.0-small.zip
    rm CheXpert-v1.0-small.zip
    cd ..
fi

##########################
# CIVIL COMMENTS DATASET #
##########################
# CHECK IF THIS IS CORRECT WITH THE OUTPUT OF THE BERT MODEL!!!!!
if [[ -f civil/all_data.csv && -f civil/train.csv && -f civil/test.csv ]]; then # We assume you have everything if you have these tree files
    echo "Civil Comments dataset found!"
else
    # Before starting you need to install kaggle with pip or anaconda
    # First you need to register and verify on Kaggle and accept the competition rules.
    # Then you need to download the API token and put it in the .kaggle folder of your user account.
    # This file is called kaggle.json
    [[ ! -d "civil" ]] &&  mkdir civil
    cd civil
    [[ ! -f jigsaw-unintended-bias-in-toxicity-classification.zip ]] && kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
    unzip -u jigsaw-unintended-bias-in-toxicity-classification.zip
    rm jigsaw-unintended-bias-in-toxicity-classification.zip
    cd ..
fi

##################
# CELEBA DATASET #
##################
CELEB_URL="'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'"
CELEB_ANNO_URL = "'https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U'"
CELEB_EVAL_URL = "'https://drive.google.com/uc?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk'"
if [[ -d celeba/img_align_celeba ]] && [[-d celeba/anno]]; then
if [[ -d celeba/img_align_celeba ]]; then
    echo "CelebA dataset found!"
else
    [[ ! -d "celeba" ]] &&  mkdir celeba
    cd celeba
    # Datasets were available on Google Drive, therefor gdown was used to import the datasets
    [[ ! -f "img_align_celeba.zip" ]] && python -c "import gdown; gdown.download($CELEB_URL, '.')"
    unzip -u img_align_celeba.zip
    rm img_align_celeba.zip
    cd .. 
    [[ ! -d "anno"]] && mkdir anno
    cd anno 
    [[ ! -f "list_atr_celeba.txt" ]] && python -c "import gdown; gdown.download($CELEB_ANNO_URL, '.')"
    [[ ! -f "list_eval_partition.txt" ]] && python -c "import gdown; gdown.download($CELEB_EVAL_URL, '.')"
fi
