# models/linear_regression.py
class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = []

    def fit(self, x, y):
        self.weights = [0.0] * len(x[0])
        for _ in range(self.iterations):
            for i in range(len(x)):
                prediction = self.predict(x[i])
                error = y[i] - prediction
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * x[i][j]

    def predict(self, x):
        return sum(w * x for w, x in zip(self.weights, x))
