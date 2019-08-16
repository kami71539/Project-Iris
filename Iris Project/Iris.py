#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv('iris.data')
X=dataset.iloc[ : , :-1].values
y=dataset.iloc[ : , -1].values

#Encoding the variable
from sklearn.preprocessing import LabelEncoder
Label_Encoder=LabelEncoder()
y=Label_Encoder.fit_transform(y)

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fitting the model to the dataset
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train, y_train)

#Prediciting the data
y_pred=classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))


#Applying Cross Validation Score
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X_train, y_train, scoring='accuracy')
accuracies.mean()
accuracies.std()

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score