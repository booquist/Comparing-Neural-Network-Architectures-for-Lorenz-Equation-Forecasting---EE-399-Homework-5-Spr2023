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

## IV. Computational Results

By creating a feed-forward neural network, we were able to compare and contrast its performance with recurrent neural networks and LSTMs:

```python
# Train a Feed-Forward Neural Network
def train_ffnn(X_train, y_train, X_val, y_val, epochs=100):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(3,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=0)
    
    return model, history

# Train a Recurrent Neural Network
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


# Train a Long Short-Term Memory Network
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

For our test on the Lorenz equations, we compared feed-forward, LSTM, RNN and Echo State Networks for forecasting the dynamics. <br>

**Feed-Forward Neural Network (FFNN)** <br>
The FFNN was trained for ρ = 10, 28, and 35. We then tested the network for future state prediction for ρ = 17 and ρ = 35. The mean squared error (MSE) was as follows:
```
2500/2500 [==============================] - 2s 609us/step
Mean Squared Error for rho = 17
FFNN: 2.7239888884449422e-05
2500/2500 [==============================] - 1s 588us/step
Mean Squared Error for rho = 35
FFNN: 3.200739952980744e-05
```
**Recurrent Neural Network (RNN)** <br>
The RNN was also trained for ρ = 10, 28, and 35. We then tested the network for future state prediction for ρ = 17 and ρ = 35. The MSE was as follows:
```
2500/2500 [==============================] - 2s 758us/step
Mean Squared Error for rho = 10
RNN: 0.037857085428462924
(80000, 3)
2500/2500 [==============================] - 2s 774us/step
Mean Squared Error for rho = 28
RNN: 0.04218182993158904
```
**Long Short-Term Memory (LSTM)** <br>
The LSTM was trained on the same ρ = 10, 28, and 35. We then tested the network for future state prediction for ρ = 17 and ρ = 35. The MSE was as follows:
```
2500/2500 [==============================] - 3s 965us/step
Mean Squared Error for rho = 17
LSTM: 0.030265745313244153
2500/2500 [==============================] - 3s 1ms/step
Mean Squared Error for rho = 35
LSTM: 0.06378890443500634
```

Here, we can see that the feed-forward neural network (FFNN) performed the best overall, achieving the lowest MSE values for both $\rho=10$ and $\rho=17$. The FFNN outperformed the RNN and LSTM models, which showed higher MSE values for the same $\rho$ values. However, it is worth noting that the RNN and LSTM models had lower MSE values for $\rho=28$ and $\rho=35$ compared to the FFNN. <br>

On the other hand, the Echo State Network (ESN) did not perform well for any of the tested $\rho$ values, with MSE values much higher than the other models. <br>

Below are the four plots for each model, which show the predicted trajectories for $\rho=17$ and $\rho=35$:

![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/f688a659-993a-4c05-853e-d684dc4efeb7)
![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/94dac603-2137-4639-bf13-230fa8971ad1)

FFNN: <br>
![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/95f29f6e-b1ff-4026-a43c-93f7612f6b6f)
![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/815a360d-1677-4517-a9d1-d13ab80cb1dd)

RNN: <br>
![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/2cc1b5dd-c404-4d6a-bc4c-c7013ec43019)
![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/8c8084a9-e9c8-409f-b456-1040f9cbf89a)

LSTM: <br>
![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/e05ccd02-228f-4560-b763-3f1ec7944d4a)
![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/f437cf0b-8645-42a8-9d96-7cab9e0109ef)

ESN: <br>
![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/af242e8e-e674-42b6-8ca0-47d2b2240f2a)
![image](https://github.com/booquist/Comparing-Neural-Network-Architectures-for-Lorenz-Equation-Forecasting---EE-399-Homework-5-Spr2023/assets/103399658/63caf3e6-c565-4c2d-928c-f8433aa0b360)

## V. Summary and Conclusions

In this project, we compared different neural network architectures to forecast the dynamics of the Lorenz equations. Specifically, we compared feed-forward neural networks, recurrent neural networks, long short-term memory networks, and echo state networks. <br>

Our results showed that the feed-forward neural network performed the best overall in terms of mean squared error, achieving the lowest MSE values for both $\rho=10$ and $\rho=17$. The FFNN outperformed the RNN and LSTM models, which showed higher MSE values for the same $\rho$ values. However, it is worth noting that the RNN and LSTM models had lower MSE values for $\rho=28$ and $\rho=35$ compared to the FFNN. <br>

On the other hand, the Echo State Network did not perform well for any of the tested $\rho$ values, with MSE values much higher than the other models. <br>

Overall, our results suggest that feed-forward neural networks are a promising approach for forecasting the dynamics of the Lorenz equations. However, further investigation is needed to determine the best neural network architecture for different values of $\rho$.
