# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Reading the dataset
dataset = pd.read_csv('breast_cancer.csv')
# index locating
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test results
y_pred = classifier.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Model accuracy
acc = accuracy_score(y_test, y_pred)
print(acc)

# K-fold cross validation
accuracies_k = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies_k.mean() * 100))
print("Standard deviation : {:.2f} %".format(accuracies_k.std() * 100))
