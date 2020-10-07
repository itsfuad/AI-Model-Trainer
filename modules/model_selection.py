# modules/model_selection.py
import random

def k_fold_split(data, k=5):
    random.shuffle(data)
    folds = [data[i::k] for i in range(k)]
    return folds

def train_test_split(data, test_ratio=0.2):
    test_size = int(len(data) * test_ratio)
    train_data = data[:-test_size]
    test_data = data[-test_size:]
    return train_data, test_data
