import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import pickle
import joblib

iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns=['sepal length','sepal width','petal length','petal width'])
df['target'] = iris['target']
#print(df.head())

train, test = train_test_split(df, test_size = 0.3, random_state = 101)

X = train.drop(['target'], axis=1)
y = train['target']

X1 = test.drop(['target'], axis=1)
y1 = test['target']
model_lr = LogisticRegression(class_weight = 'balanced')

model_lr.fit(X,y)

y_pred = model_lr.predict(X)
train_accuracy = accuracy_score(y, y_pred)
y_pred1 = model_lr.predict(X1)
test_accuracy = accuracy_score(y1, y_pred1)

print(train_accuracy, test_accuracy)

joblib.dump(model_lr, 'iris_lr.pkl')

