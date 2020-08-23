# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Reading the dataset
dataset = pd.read_csv('breast_cancer.csv')
# index locating
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Training the model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
