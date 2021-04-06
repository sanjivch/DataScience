import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, optimizers, losses, metrics
import keras
from sklearn.model_selection import train_test_split

# ====================================================================================
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


def normalize_values(_var):
    ''' Normalize the variable between
        O and 1.
    '''
    return (_var-_var.min())/(_var.max()-_var.min())

def generate_target(viscosity, horizontal_permeability, k_v_h, porosity, steam_injection_pressure, steam_injection_rate, steam_quality):
    ''' Define arbitrary relation between the
        features and target.
    '''
    recovery = 0.5*(steam_quality**3)\
           -0.25*(steam_injection_pressure**1.8)\
           -0.15*(steam_injection_rate**1.75) \
           +0.35 * viscosity \
           +0.41 * horizontal_permeability \
           +0.15 * k_v_h \
           -0.075 * (porosity ** 1.25) \
           +np.random.rand() # random unmeasured noise

    return recovery

def compare_results(prediction, actual):
    plt.plot(prediction, '*', label='Prediction')
    plt.plot(actual, '.', label='Actual')
    plt.legend(loc='best')
    plt.ylabel('Normalized Recovery Rate')
    plt.xlabel('Sample #')
    plt.show()

def plot_metrics(_model):
    # Loss
    print(_model.history)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Base Model')
    ax1.plot(_model.history['loss'])
    ax1.plot(_model.history['val_loss'])
    ax1.set_title('Training and Validation -  Loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['training', 'validation'], loc='best')

    # summarize history for MSE
    ax2.plot(_model.history['mean_squared_error'])
    ax2.plot(_model.history['val_mean_squared_error'])
    ax2.set_title('Training and Validation - MSE')
    ax2.set_ylabel('MSE')
    ax2.set_xlabel('epoch')
    ax2.legend(['training', 'validation'], loc='best')
    plt.show()


# =========================================================================================================
# Training data
#
# The data reference is taken from
# "Predicting the performance of steam assisted gravity drainage (SAGD)
# method utilizing artiﬁcial neural network (ANN)" by
# Areeba Ansari, Marco Heras, Julianne Nones, Mehdi Mohammadpoor, Farshid Torab
#
# -----------------------------------------------------------------
# InputParameter - Minimum | Maximum | Mean | Standard Deviation |
# -----------------------------------------------------------------
# Steam quality (w of dry steam/w of dry + wet steam) - 0.68 0.88 0.8 0.05
# Steam Injection Pressure (kPa) - 1200 9480 3012 1435
# Steam Injection Rate (bbl/day) -  31.447 26952.12872 5980 6091
# Oil Viscosity (cP) - 1965 5000000 1039965 965149
# Horizontal Permeability (mD) -  1000 10000 4295 1668
# k k v h / - 0.1000 1.0000 0.5927 0.2118
# Porosity (%) - 20.00 40.00 32.27 3.72
# Pay Thickness (ft) - 0.41 525 87 87
# -----------------------------------------------------------------
# OutputParameter
# -----------------------------------------------------------------
# Recovery Factor (%)
# -----------------------------------------------------------------


NUM_SAMPLES = 2000

viscosity = generate_data(1965, 5000000, 1039965, 965149, NUM_SAMPLES)
horizontal_permeability = generate_data(1000, 10000, 4295, 1668, NUM_SAMPLES)
k_v_h = generate_data(0.1000, 1.0000, 0.5927,  0.2118, NUM_SAMPLES)
porosity = generate_data(20.00, 40.00, 32.27, 3.72, NUM_SAMPLES)
pay_thickness = generate_data(0.41, 525, 87, 87, NUM_SAMPLES)
steam_injection_pressure = generate_data(1200, 9480, 3012, 1435, NUM_SAMPLES)
steam_injection_rate = generate_data(31.447, 26952.12872, 5980, 6091, NUM_SAMPLES)
steam_quality = generate_data(0.68, 0.88, 0.8, 0.05, NUM_SAMPLES)
recovery_rate = generate_target(viscosity, horizontal_permeability, k_v_h, porosity, steam_injection_pressure, steam_injection_rate, steam_quality)
print(recovery_rate)
# Normalize the values
viscosity = normalize_values(viscosity)
horizontal_permeability = normalize_values(horizontal_permeability)
k_v_h = normalize_values(k_v_h)
porosity = normalize_values(porosity)
pay_thickness = normalize_values(pay_thickness)
steam_injection_pressure = normalize_values(steam_injection_pressure)
steam_injection_rate = normalize_values(steam_injection_rate)
recovery_rate = normalize_values(recovery_rate)

# Get inputs and target variable.
# Split them in 75:25 ratio
# Train:test = 0.75:0.25
TEST_SIZE = 0.25
SPLIT_CUTOFF = int(NUM_SAMPLES*(1-TEST_SIZE))

X = np.array([viscosity, horizontal_permeability, k_v_h, porosity, pay_thickness, steam_injection_rate, steam_injection_pressure, steam_quality])
y = recovery_rate

X = X.reshape(NUM_SAMPLES,8)
y = y.reshape(NUM_SAMPLES)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=101)

print('X_train Shape:\t{}\ny_train Shape:\t{}'.format(X_train.shape, y_train.shape))
print('X_test Shape:\t{}\ny_test Shape:\t{}'.format(X_test.shape, y_test.shape))

# Define network/model
inputs = keras.Input(shape=(8,))
#print(inputs.shape, inputs.dtype)

layer_1 = layers.Dense(12, activation = 'relu')
x_1 = layer_1(inputs)
#print(x_1.shape)

layer_2 = layers.Dense(16, activation= 'relu')(x_1)
outputs = layers.Dense(1)(layer_2)
#print(outputs.shape)

model =  keras.Model(inputs=inputs,
                     outputs=outputs,
                     name='base_model')
model.summary()

# Model Parameters
# =========================================================================
LEARNING_RATE = 0.0001
LOSS = losses.MeanSquaredError()
OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE)
METRICS = [metrics.MeanSquaredError()]
BATCH_SIZE = 32
NUM_EPOCHS = 250
VALIDATION_SPLIT = 0.2

model.compile(loss=LOSS,
              optimizer=OPTIMIZER,
              metrics=METRICS)

# Fit the model on the training set
base_model = model.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_split=VALIDATION_SPLIT)

# Evaluate the model on the test set
test_scores = model.evaluate(X_test,
                             y_test,
                             batch_size= BATCH_SIZE,
                             verbose=2)

print("Test loss:\t{:.2f}\nTest MSE:\t{:.2f}".format(test_scores[0], test_scores[1]))

# Predict on the test data
y_pred = model.predict(X_test)

# Plot results and metrics
plot_metrics(base_model)
compare_results(y_pred, y_test)