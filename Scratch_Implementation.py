# Import Numpy
import numpy as np

# For reproducibility
np.random.seed(0)

# create a toy dataset
X = np.random.rand(4,2)
y = np.array([[0], [1], [1], [0]])

# Initalise parameters
def init_params(input_neurons,hidden_neurons,output_neurons):
    np.random.seed(0)
    W1 = np.random.rand(hidden_neurons,input_neurons)
    B1 = np.random.rand(hidden_neurons,1)
    W2 = np.random.rand(output_neurons,hidden_neurons)
    B2 = np.random.rand(output_neurons,1)
    parameters = {"W1":W1, "B1":B1, "W2":W2, "B2":B2}
    
    return parameters

# Define activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define forward prop
def forward_prop(X,parameters):
    W1,B1,W2,B2 = parameters.values()

    # Compute activations of hidden layer
    Z1 = np.dot(W1,X.T) + B1
    A1 = sigmoid(Z1)

    # Compute activations of output layer
    Z2 = np.dot(W2,A1) + B2
    A2 = sigmoid(Z2)

    # Store intermediate values later used for back prop
    intermed_vals = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}

    return A2, intermed_vals

# Compute loss function
def binary_cross_entropy_loss(A2,y):
    m = y.shape[0]
    loss = (-1/m) * np.sum(y * np.log(A2) + (1-y) * np.log(1-A2))
    return loss

# Back Propogation
def back_prop(parameters,intermed_vals,X,y):

    m = y.shape[0]

    Z1 = intermed_vals["Z1"]
    A1 = intermed_vals["A1"]
    Z2 = intermed_vals["Z2"]
    A2 = intermed_vals["A2"]

    # Compute derivative of loss with respect to A2
    dA2 = -(y.T/A2) + (1-y.T)/(1-A2)

    # Compute derivative of Z2
    dZ2 = dA2 * (A2 * (1-A2))

    # Compute derivatives of weights and bias in outer layer
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    dB2 = (1/m) * np.sum(dZ2)

    # Compute the derivatives of A1 and Z1 with respect to the loss
    dA1 = np.dot(parameters["W2"].T,dZ2)
    dZ1 = dA1 * A1 * (1-A1)

    # Compute derivatives of weights and bias in hidden layer
    dW1 = (1/m) * np.dot(dZ1,X) 
    dB1 = (1/m) * np.sum(dZ1)

    gradients = {"DW1":dW1, "DB1":dB1, "DW2":dW2, "DB2":dB2}
    return gradients

# Update parameters
def update_params(parameters,gradients,learning_rate):
    # Retrieve Gradients
    dW1 = gradients["DW1"]
    dB1 = gradients["DB1"]
    dW2 = gradients["DW2"]
    dB2 = gradients["DB2"]
    
    # Retrieve parameters
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]

    # Update
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2

    parameters = {"W1":W1, "B1":B1, "W2":W2, "B2":B2}
    return parameters

# Train neural network
def train(X,y,hidden_layer_size, num_iterations, learning_rate):
    
    parameters = init_params(X.shape[1],hidden_layer_size,1)
    
    for i in range(num_iterations):
        # Forward prop
        A2, intermed_vals = forward_prop(X,parameters)
        
        # Loss
        loss = binary_cross_entropy_loss(A2,y)

        # Back Prop
        gradients = back_prop(parameters,intermed_vals,X,y)

        # Update Params
        parameters = update_params(parameters,gradients,learning_rate)
        
        if i % 50 == 0:
            print(f"Iteration {i}, Loss = {loss}")

    # Return optimal parameters
    return parameters

# Predictions
def predict(X,parameters):
    predictions = forward_prop(X,parameters)
    return predictions  

# Compute predictions
parameters = train(X,y,hidden_layer_size = 3,num_iterations = 1000,learning_rate = 0.01)
predictions,_ = predict(X,parameters)
predictions > 0.5