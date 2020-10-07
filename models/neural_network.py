# models/neural_network.py
import math

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, iterations=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [random.uniform(-1, 1) for _ in range(hidden_size)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        for _ in range(self.iterations):
            for i in range(len(X)):
                hidden_layer_input = [sum(X[i][j] * self.weights_input_hidden[j][k] for j in range(self.input_size)) for k in range(self.hidden_size)]
                hidden_layer_output = [self.sigmoid(x) for x in hidden_layer_input]
                output_layer_input = sum(hidden_layer_output[k] * self.weights_hidden_output[k] for k in range(self.hidden_size))
                output = self.sigmoid(output_layer_input)

                output_error = y[i] - output
                output_delta = output_error * self.sigmoid_derivative(output)

                hidden_error = [output_delta * self.weights_hidden_output[k] for k in range(self.hidden_size)]
                hidden_delta = [hidden_error[k] * self.sigmoid_derivative(hidden_layer_output[k]) for k in range(self.hidden_size)]

                for k in range(self.hidden_size):
                    self.weights_hidden_output[k] += self.learning_rate * output_delta * hidden_layer_output[k]
                for j in range(self.input_size):
                    for k in range(self.hidden_size):
                        self.weights_input_hidden[j][k] += self.learning_rate * hidden_delta[k] * X[i][j]

    def predict(self, X):
        hidden_layer_input = [sum(X[j] * self.weights_input_hidden[j][k] for j in range(self.input_size)) for k in range(self.hidden_size)]
        hidden_layer_output = [self.sigmoid(x) for x in hidden_layer_input]
        output_layer_input = sum(hidden_layer_output[k] * self.weights_hidden_output[k] for k in range(self.hidden_size))
        return self.sigmoid(output_layer_input)
