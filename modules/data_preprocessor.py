# modules/data_preprocessor.py
def normalize_data(data):
    features = zip(*data)
    normalized_features = []
    for feature in features:
        min_val = min(feature)
        max_val = max(feature)
        normalized_feature = [(x - min_val) / (max_val - min_val) for x in feature]
        normalized_features.append(normalized_feature)
    return list(zip(*normalized_features))

def add_bias_term(data):
    return [[1.0] + row for row in data]
