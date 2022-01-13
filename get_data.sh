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

##########################
# CIVIL COMMENTS DATASET #
##########################
# CHECK IF THIS IS CORRECT WITH THE OUTPUT OF THE BERT MODEL!!!!!
if [[ -f civil/train.csv && -f civil/valid.csv && -d civil/train && civil/valid ]]; then
    echo "Civil Comments dataset found!"
else
    # Before starting you need to install kaggle with pip or anaconda
    # First you need to register and verify on Kaggle and accept the competition rules.
    # Then you need to download the API token and put it in the .kaggle folder of your user account.
    # This file is called kaggle.json
    [[ ! -d "civil" ]] &&  mkdir civil
    cd civil
    kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
    cd ..
fi

##################
# CELEBA DATASET #
##################
# CHECK IF THIS IS CORRECT WITH THE OUTPUT OF THE BERT MODEL!!!!!
if [[ -f celeba/train.csv && -f celeba/valid.csv && -d celeba/train && celeba/valid ]]; then
    echo "CelebA dataset found!"
else
    [[ ! -d "celeba" ]] &&  mkdir celeba
    cd celeba
     
fi
