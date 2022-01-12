#!/bin/bash

cd data

# ADULT DATASET
# Make the directory
[[ ! -d "adult" ]] &&  mkdir adult

# Download the files
cd adult
[[ ! -f "adult.data" ]] && wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
[[ ! -f "adult.test" ]] && wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
cd ..


# CHEXPERT DATASET
URL="" # See README.md

# Make the directory
[[ ! -d "chexpert" ]] &&  mkdir chexpert
# Download the files
cd chexpert
[[ ! -f "CheXpert-v1.0-small.zip" ]] && wget $URL
cd ..

# CIVIL COMMENTS DATASET
# Before starting you need to install kaggle with pip or anaconda
# First you need to register and verify on Kaggle and accept the competition rules.
# Then you need to download the API token and put it in the .kaggle folder of your user account.
# This file is called kaggle.json
[[ ! -d "Civil" ]] &&  mkdir civil
cd civil
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
