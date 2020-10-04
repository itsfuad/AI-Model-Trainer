# train_linear_regression.py
from modules.data_loader import load_data, split_data
from modules.data_preprocessor import normalize_data, add_bias_term
from models.linear_regression import LinearRegression
from modules.metrics import mean_squared_error

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

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = [model.predict(x) for x in x_test]
    error = mean_squared_error(y_test, predictions)

    print(f'Mean Squared Error: {error}')

if __name__ == "__main__":
    main()
