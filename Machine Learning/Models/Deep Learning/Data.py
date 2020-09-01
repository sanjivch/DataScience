import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
import keras

'''
Parameter Minimum Maximum Mean Standard Deviation
Input
----- 
Oil Viscosity (cP) 1965 5000000 1039965 965149
Horizontal Permeability (mD) 1000 10000 4295 1668
k k v h / 0.1000 1.0000 0.5927 0.2118
Porosity (%) 20.00 40.00 32.27 3.72
Pay Thickness (ft) 0.41 525 87 87
Steam Injection Pressure (kpa) 1200 9480 3012 1435
Steam Injection Rate (bbl/day) 31.447 26952.12872 5980 6091

Output
------ 
Recovery Factor (%) 23 88 69 14
'''
# Higher viscosity will increase oil recovery rate

def generate_data(min_value, max_value, mean, std, num_samples, random_seed = 100):

    # Set seed to maintain consistency
    np.random.seed(random_seed)

    # Scale across the range of variable
    scale = max_value - min_value
    location = min_value

    # Mean and standard deviation of the unscaled beta distribution
    unscaled_mean = (mean - min_value) / scale
    unscaled_var = (std / scale) ** 2

    # Computation of alpha and beta can be derived from mean and variance formulas
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t ** 2) - (2 * t) - 1) / ((t ** 3) + (3 * t ** 2) + (3 * t) + 1)
    alpha = beta * t

    # Not all parameters may produce a valid distribution
    if alpha <= 0 or beta <= 0:
        raise ValueError('Cannot create distribution for the given parameters.')

    # Make scaled beta distribution with computed parameters
    distribution = scipy.stats.beta(alpha, beta, scale=scale, loc=location)

    return distribution.rvs(size=num_samples)

def check_data(_var):
    print(_var)
    print('mean:',_var.mean(),'std:', _var.std(), 'min:', _var.min(), 'max:', _var.max())
    plt.plot(np.sort(_var))
    plt.show()

def normalize_values(_var):
    ''' Normalize the variable between
        O and 1.
    '''
    return (_var-_var.min())/(_var.max()-_var.min())

def generate_target(viscosity, horizontal_permeability, k_v_h, porosity, steam_injection_pressure, steam_injection_rate):
    ''' Define arbitrary relation between the
        features and target.
    '''
    return 0.35*viscosity+0.41*horizontal_permeability+0.15*k_v_h-0.075*porosity-0.25*steam_injection_pressure-0.15*steam_injection_rate+0.32

# Training data
NUM_SAMPLES = 2000
viscosity = generate_data(1965, 5000000, 1039965, 965149, NUM_SAMPLES)
horizontal_permeability = generate_data(1000, 10000, 4295, 1668, NUM_SAMPLES)
k_v_h = generate_data(0.1000, 1.0000, 0.5927,  0.2118, NUM_SAMPLES)
porosity = generate_data(20.00, 40.00, 32.27, 3.72, NUM_SAMPLES)
pay_thickness = generate_data(0.41, 525, 87, 87, NUM_SAMPLES)
steam_injection_pressure = generate_data(1200, 9480, 3012, 1435, NUM_SAMPLES)
steam_injection_rate = generate_data(31.447, 26952.12872, 5980, 6091, NUM_SAMPLES)
recovery_rate = generate_target(viscosity, horizontal_permeability, k_v_h, porosity, steam_injection_pressure, steam_injection_rate)

# Normalize the values

viscosity = normalize_values(viscosity)
horizontal_permeability = normalize_values(horizontal_permeability)
k_v_h = normalize_values(k_v_h)
porosity = normalize_values(porosity)
pay_thickness = normalize_values(pay_thickness)
steam_injection_pressure = normalize_values(steam_injection_pressure)
steam_injection_rate = normalize_values(steam_injection_rate)
recovery_rate = normalize_values(recovery_rate)


X = np.array([viscosity, horizontal_permeability, k_v_h, porosity, pay_thickness, steam_injection_rate, steam_injection_pressure])
y = recovery_rate
X_train = X[:,:1500]
y_train = y[:1500]

X_test = X[:,1500:]
y_test = y[1500:]

print(X.shape, y.shape, 'test: ',X_test.shape, y_test.shape)

# Build model
inputs = keras.Input(shape=(7,))
print(inputs.shape, inputs.dtype)

layer_1 = layers.Dense(14, activation = 'relu')
x_1 = layer_1(inputs)
print(x_1.shape)

layer_2 = layers.Dense(21, activation= 'relu')(x_1)
outputs = layers.Dense(1)(layer_2)
print(outputs.shape)

model =  keras.Model(inputs=inputs, outputs=outputs, name='base_model')
model.summary()
X_train = X_train.reshape(1500,7)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['mean_squared_error'],

)

history = model.fit(X_train, y_train, batch_size=32, epochs=500, validation_split=0.2)
X_test = X_test.reshape(500,7)
test_scores = model.evaluate(X_test, y_test, batch_size= 25, verbose=2)
print("Test loss:", test_scores[0])
print("Test MAE:", test_scores[1])

print(test_scores)
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
