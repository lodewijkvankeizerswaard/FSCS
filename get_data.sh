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
