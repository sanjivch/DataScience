import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

train_file = r"C:\Users\sanjiv\Documents\Datasets\Kaggle\House prices\train.csv"
test_file = r"C:\Users\sanjiv\Documents\Datasets\Kaggle\House prices\test.csv"
#Train file
house_data_train = pd.read_csv(train_file)



TARGET = 'SalePrice'
X_train = house_data_train.drop([TARGET], axis = 1)
y_train = house_data_train[TARGET]
#print(house_data_train.head(10))
#X_train['SalePrice'] = y_train
#print(X_train.head(10))
#X_train = pd.concat
print(X_new.head(10))
#print(y_train.head(10))
#Test file
#house_data_test = pd.read(test_file)
#X_test = house_data_test

# Model

#model_lm = LinearRegression()
#model_lm.fit(X_train,y_train)

# Prediction

#y_pred = model_lm.predict(X_test)




