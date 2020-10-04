# train_logistic_regression.py
from modules.data_loader import load_data, split_data
from modules.data_preprocessor import normalize_data, add_bias_term
from models.logistic_regression import LogisticRegression
from modules.metrics import accuracy

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

    model = LogisticRegression()
    model.fit(x_train, y_train)

    predictions = [1 if model.predict(x) > 0.5 else 0 for x in x_test]
    acc = accuracy(y_test, predictions)

    print(f'Accuracy: {acc}')

if __name__ == "__main__":
    main()
