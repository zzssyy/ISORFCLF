# ISORFCLF

A Novel Method for Identifying Small Open Reading Frames in Pri-miRNA using Class-imbalanced Learning Framework (ISORFCLF)

It consists of two parts: feature representation and a class-imbalanced learning framework. 

Firstly, feature representation is to extract different features. 

Then, a class-imbalanced learning framework is to balance data, and complete the classification task.

# Requirement

Python = 3.7

# Usage
data_processing.py is used for feature extraction and divide data

ensemble_learning.py is used to ensemble machine learning models

ipso.py is used for updating weights for different models in ensemble learning

ISORFCLF.py is used for identifying sORFs
