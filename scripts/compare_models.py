# scripts/compare_models.py
from modules.data_loader import load_data, split_data
from modules.data_preprocessor import normalize_data, add_bias_term
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.neural_network import NeuralNetwork
from models.decision_tree import DecisionTree
from models.support_vector_machine import SVM
from modules.metrics import mean_squared_error, accuracy

def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    if isinstance(model, LinearRegression):
        return mean_squared_error(y_test, predictions)
    else:
        return accuracy(y_test, predictions)

def main():
    _, data = load_data('data/dataset.csv')
    train_data, test_data = split_data(data)

    train_data = normalize_data(train_data)
    train_data = add_bias_term(train_data)
    
    test_data = normalize_data(test_data)
    test_data = add_bias_term(test_data)

    x_train = [row[:-1] for row in train_data]
    y_train = [row[-1] for row in train_data]
    
    x_test = [row[:-1] for row in test_data]
    y_test = [row[-1] for row in test_data]

    models = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(),
        "Neural Network": NeuralNetwork(input_size=len(x_train[0]), hidden_size=10, output_size=1),
        "Decision Tree": DecisionTree(max_depth=5),
        "SVM": SVM()
    }

    for name, model in models.items():
        score = evaluate_model(model, x_train, y_train, x_test, y_test)
        print(f"{name}: {score}")

if __name__ == "__main__":
    main()
