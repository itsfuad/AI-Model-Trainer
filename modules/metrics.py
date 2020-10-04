# modules/metrics.py
def mean_squared_error(y_true, y_pred):
    error = 0.0
    for true, pred in zip(y_true, y_pred):
        error += (true - pred) ** 2
    return error / len(y_true)

def accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)
