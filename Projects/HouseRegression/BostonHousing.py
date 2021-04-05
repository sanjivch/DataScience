import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers,losses,models, optimizers, Input, Model, metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

house_data = pd.read_csv(r'C:\Users\sanjiv\Documents\Datasets\Misc\boston.csv',delim_whitespace=True, header=None)
print(house_data.head())

# Normalize data
scaler = MinMaxScaler()
house_data_scaled = scaler.fit_transform(house_data.values)
X = house_data_scaled[:, 0:13]
y = house_data_scaled[:, 13]


print(X.shape, y.shape)

inputs = Input(shape=(13,))

layer_1 = layers.Dense(10, activation='relu')
x_1 = layer_1(inputs)

layer_2 = layers.Dense(6, activation='relu')
x_2 = layer_2(x_1)

layer_3= layers.Dense(1)
outputs = layer_3(x_2)

model = Model(inputs = inputs, outputs = outputs, name = 'Base_model')

X_train = X[:400, :]
X_test = X[400:, :]
y_train = y[:400, ]
y_test = y[400:, ]
print(X_train.shape, y_train.shape)
print(model.summary())

OPTIMIZER = optimizers.Adam(learning_rate=0.0001)
LOSS = 'mean_squared_error'
METRICS = [metrics.MeanSquaredError()]
#metrics.
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS,)

history = model.fit(X_train, y_train, batch_size=15, epochs = 500, validation_split=0.3)

test_scores = model.evaluate(X_test, y_test, verbose=2)

print('Test loss: {:.2f}\nTest MAE: {:.2f}'.format(test_scores[0], test_scores[1]))

y_pred = model.predict(X_test)
print(y_pred, y_test)

print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for MAE
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()