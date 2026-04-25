# Evaluating_Generalizability_For_Hypoglycemia_Classification_Using_DiaData

This repository evaluates the generalizability of population-based and age-segmented models for hypoglycemia classification using DiaData, a large integrated CGM Dataset of subjects with Type 1 Diabetes presented in: https://github.com/Beyza-Cinar/DiaData. Detailed instruction on acquiring the single datasets and restricted datasets is provided in https://github.com/Beyza-Cinar/DiaData as well. The raw and preprocessed datasets can be downloaded from [https://zenodo.org/records/16875703](https://zenodo.org/records/17285631).

We investigate: 1) the performance of a Fully Convolutional Network (FCN) trained on the whole population and tested separately on subjects across four age groups, 2) the impact of age-segmented models, where the same model architecture is trained separately for the age groups, and 3) the effect of individualization through transfer learning, where the model is fine-tuned with the test subjects’ data.

## Reference

The paper associated with this code will be published at the IEEE CAI 2026 Conference. 

## Requirements

python version 3.11.4

matplotlib version 3.7.2

pandas version 2.0.3

numpy version 1.24.3

polars version 1.27.1

TensorFlow version 2.13.0

## Code Organization

The code is organized as follows:

The Code folder includes the scripts
- reading the dataset (Data_read.ipynb),
- creating the age groups for age segmented model training (Age_Groups.ipynb),
- conducting ablation studies with 5 and 15 min sampling (Ablation_Studies_5min_Sampling.ipynb, Ablation_Studies_15min_Sampling.ipynb)
- training the base models, testing and fine-tuning the models, and evaluating the models for global population based, age-segmented, and individualized approaches (Training_Models.ipynb, Testing_Models.ipynb, Evaluating_Models.ipynb)
- conducting statistical significance testing (Statistical_Significance.ipynb)
- helping functions in python scripts (data_integration.py, data_models.py, data_preprocessing.py)
