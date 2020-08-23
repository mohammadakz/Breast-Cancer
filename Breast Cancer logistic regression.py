# Importing libraries
import pandas as pd

# Reading the dataset
dataset = pd.read_csv('breast_cancer.csv')
# index locating
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
