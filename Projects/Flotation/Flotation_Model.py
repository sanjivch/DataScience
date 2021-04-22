import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load data
csv_file = r'C:\Users\sanjiv\Documents\Datasets\Kaggle\Flotation\MiningProcess_Flotation_Plant_Database.csv'
master = pd.read_csv(csv_file, decimal=',', parse_dates=['date'], index_col='date')
#master = pd.read_csv(csv_file, decimal=',')

st.title('Flotation model')
print(master.head())
# # See if there is a sesonality per month or a per shift basis
# master['month'] = master.index.month
# master['hour'] = master.index.hour
# monthly_data = master.resample('M').mean()
# print(monthly_data.head())
# daily_data = master.resample('D').mean()
#
# sns.lineplot(x=daily_data.index, y='% Silica Concentrate', data=daily_data)
# plt.show()

plt.plot(master["% Iron Concentrate"])
matrix = np.triu(master.corr())
sns.heatmap(master.corr(), annot = True, fmt='.1g', linewidths=1, mask = matrix)
plt.show()
print(master.corr())
master = master.drop(['date'], axis=1)#, '% Iron Concentrate', 'Ore Pulp pH', 'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow'], axis=1)

X = master.drop(['% Silica Concentrate'], axis =1)
y = master['% Silica Concentrate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
#st.selectbox('Model selection', ['Linear Regression', 'SGD'])


model_sgd = SGDRegressor()
model_sgd.fit(X_train, y_train)

y_pred_sgd = model_sgd.predict(X_test)
sns.scatterplot(x=y_test,y= y_pred_lr)
sns.scatterplot(x= y_test, y = y_pred_sgd)
plt.show()
