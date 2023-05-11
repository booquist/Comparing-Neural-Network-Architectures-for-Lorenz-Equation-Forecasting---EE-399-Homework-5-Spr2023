# Exploring Neural Networks for Solving Lorenz Equations and Comparing with Echo State Networks

**Author:** Brendan Oquist <br>
**Abstract:** This project focuses on the application of feedforward neural networks (FFNN) and Echo State Networks (ESN) for predicting the behavior of the Lorenz equations as part of EE 399 Spring Quarter 2023. The task involves fitting a FFNN and an ESN model to the data generated using the Lorenz equations for different rho values and comparing the performance of these models using the Mean Squared Error (MSE) metric.

## I. Introduction and Overview
In this project, we use the Lorenz equations to generate training and testing data for different rho values. We then fit a feedforward neural network and an Echo State Network to the data and compare the performance of these models using the Mean Squared Error (MSE) metric.

## II. Theoretical Background

In this section, we provide the necessary mathematical background for solving the Lorenz equations, fitting feedforward neural networks (FFNNs) and Echo State Networks (ESNs) to the generated data, and discussing the performance metrics used for evaluating the models.

### Lorenz Equations

The Lorenz equations are a system of three coupled, first-order, nonlinear differential equations that describe the behavior of a simplified fluid convection model. The equations are given by:

dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz


where `x`, `y`, and `z` are state variables, and `σ`, `ρ`, and `β` are parameters. These equations exhibit a wide range of behaviors, including sensitive dependence on initial conditions, as the parameters are varied.

### Feedforward Neural Networks (FFNNs)

Feedforward neural networks (FFNNs) are a type of artificial neural network where information flows in a forward direction from the input layer through hidden layers to the output layer. The primary components of an FFNN are the neurons, which perform a weighted sum of their inputs followed by a nonlinear activation function. The weights are adjusted during training using an optimization algorithm, such as gradient descent, to minimize the error between the predicted and actual outputs.

### Echo State Networks (ESNs)

Echo State Networks (ESNs) are a type of recurrent neural network (RNN) characterized by a large, fixed, and randomly connected reservoir of neurons. The weights connecting the input and reservoir neurons, as well as the weights within the reservoir, are not trained. Only the weights connecting the reservoir to the output neurons are trained using a linear regression technique, making ESNs computationally efficient compared to other RNNs. ESNs are particularly suited for time series prediction tasks due to their inherent memory and ability to capture long-range dependencies.

### Performance Metrics

To evaluate the performance of the FFNN and ESN models, we use the Mean Squared Error (MSE) metric, which measures the average squared difference between the predicted and actual values. The MSE is given by:

MSE = (1/n) Σ (y_i - ŷ_i)^2


where `y_i` are the actual values, `ŷ_i` are the predicted values, and `n` is the number of samples. Lower MSE values indicate better model performance.


...

## III. Algorithm Implementation and Development
In this section, we provide an overview of the code and steps taken to fit a feedforward neural network and an Echo State Network on the data generated using the Lorenz equations.

**Generating the Data and Preprocessing** <br>
We start by importing the necessary libraries, generating the data using the Lorenz equations for different rho values, and preprocessing the data using the MinMaxScaler.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate data using the Lorenz equations
...
```

**Define the Lorenz system's derivative function** <br>
```python 
def lorenz_deriv(x_y_z, t0, sigma=10, beta=8/3, rho=28):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
```
**Create function to generate training data for rho = 10, 28, 40** <br>
```python 
# Generate data for a given rho value
def generate_data(rho, seed=123):
    dt = 0.01
    T = 8
    t = np.arange(0, T + dt, dt)

    np.random.seed(seed)
    x0 = -15 + 30 * np.random.random((100, 3))

    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])

    nn_input = np.zeros((100 * (len(t) - 1), 3))
    nn_output = np.zeros_like(nn_input)

    for j in range(100):
        nn_input[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, :-1, :]
        nn_output[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, 1:, :]

    return nn_input, nn_output
```
**Train a Feed-Forward Neural Network** <br>
```python 
def train_ffnn(X_train, y_train, X_val, y_val, epochs=100):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(3,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=0)
    
    return model, history
```

**Train a Recurrent Neural Network** <br>
```python 
def train_rnn(X_train, y_train, X_val, y_val, epochs=100):
    model = Sequential()
    model.add(SimpleRNN(64, activation='tanh', input_shape=(1, 3), return_sequences=True))
    model.add(SimpleRNN(64, activation='tanh'))
    model.add(Dense(3))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Reshape the validation data
    X_val_rnn = X_val.reshape(-1, 1, 3)
    y_val_rnn = y_val.reshape(-1, 1, 3)
    
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val_rnn, y_val_rnn), verbose=0)

    return model, history
```

**Train a Long Short-Term Memory Network** <br>
```python 
def train_lstm(X_train, y_train, X_val, y_val, epochs=100):
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=(1, 3), return_sequences=True))
    model.add(LSTM(64, activation='tanh'))
    model.add(Dense(3))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Reshape the validation data
    X_val_lstm = X_val.reshape(-1, 1, 3)
    y_val_lstm = y_val.reshape(-1, 1, 3)
    
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val_lstm, y_val_lstm), verbose=0)

    return model, history
```

**Scale the data** <br>
```python 
scaler = MinMaxScaler(feature_range=(-1, 1))
```

**Create our ESN model for comparison** <br>
```python 
def create_esn_model(input_shape, units, connectivity=0.1, leaky=1, spectral_radius=0.9):
    inputs = tf.keras.Input(shape=input_shape)
    esn_outputs = tfa.layers.ESN(units, connectivity, leaky, spectral_radius)(inputs)
    output = tf.keras.layers.Dense(1)(esn_outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
    return model
```

**Train ESN** <br>
```python 
def train_esn(X_train, y_train, X_test, y_test, input_shape, reservoir_size, epochs=50, batch_size=32):
    esn_model = create_esn_model(input_shape, reservoir_size)
    esn_history = esn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)
    return esn_model, esn_history
```

**Evaluate the ESN** <br>
```python 
for rho in rho_test_values:
    X_test, y_test = generate_data(rho)

    X_test_scaled = scaler.transform(X_test)
    y_test_scaled = scaler.transform(y_test)
    X_test_esn = X_test_scaled.reshape(-1, 1, 3)

    esn_pred = esn_model.predict(X_test_esn)
    esn_mse = np.mean((y_test_scaled - esn_pred)**2)

    print(f"Mean Squared Error for rho = {rho}")
    print(f"ESN: {esn_mse}")
```
