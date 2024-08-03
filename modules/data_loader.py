# modules/data_loader.py
import csv

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            data.append([float(x) for x in row])
    return header, data

def split_data(data, train_ratio=0.9):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data
