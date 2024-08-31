import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (4 samples, 3 features)
X = np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]])

# Output dataset (4 samples, 1 output)
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed for reproducibility
np.random.seed(1)

# Initialize weights randomly with mean 0
input_layer_neurons = X.shape[1]  # Number of features in input
hidden_layer_neurons = 4          # Number of neurons in hidden layer
output_neurons = 1                # Number of neurons in output layer

# Weights
weights_input_hidden = np.random.randn(input_layer_neurons, hidden_layer_neurons)
weights_hidden_output = np.random.randn(hidden_layer_neurons, output_neurons)

# Training the network
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(final_layer_input)

    # Calculate error
    error = y - predicted_output
    if (epoch % 1000) == 0:
        print(f"Error at epoch {epoch}: {np.mean(np.abs(error))}")

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

# Print final predictions
print("Final predicted output:")
print(predicted_output)
