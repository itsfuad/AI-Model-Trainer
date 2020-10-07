# scripts/train_svm.py
from modules.data_loader import load_data, split_data
from modules.data_preprocessor import normalize_data
from models.support_vector_machine import SVM
from modules.metrics import accuracy

def main():
    header, data = load_data('data/dataset.csv')
    train_data, test_data = split_data(data)

    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    X_train = [row[:-1] for row in train_data]
    y_train = [row[-1] for row in train_data]
    
    X_test = [row[:-1] for row in test_data]
    y_test = [row[-1] for row in test_data]

    model = SVM()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy(y_test, predictions)

    print(f'Accuracy: {acc}')

if __name__ == "__main__":
    main()
