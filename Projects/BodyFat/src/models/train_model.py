import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load data
DATA = r"C:\Users\sanjiv\Documents\DataScience\Projects\BodyFat\data\raw\560_bodyfat.tsv.gz"
df = pd.read_csv(DATA, compression='gzip',sep='\t')

#print(df.isnull().mean())

# Baseline model

train_set, test_set = train_test_split(df, test_size = 0.3, random_state=101)
print(train_set.head())

X_train = train_set.drop(['target'], axis=1)
y_train = train_set['target']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
#y_train = scaler.fit_transform(y_train)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

print(X_train)


