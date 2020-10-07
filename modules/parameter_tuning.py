# modules/parameter_tuning.py
def grid_search(model, param_grid, X, y):
    best_params = None
    best_score = float('-inf')
    for params in param_grid:
        model.set_params(**params)
        model.fit(X, y)
        score = model.evaluate(X, y)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score
