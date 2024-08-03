# AI Trainer

## Overview
This project is an AI Trainer that builds, trains, and evaluates machine learning models from scratch without using third-party libraries.

## Project structure
```
ai_trainer/
├── data/
│   ├── dataset.csv
├── models/
│   ├── decision_tree.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── neural_network.py
│   └── support_vector_machine.py
├── modules/
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── metrics.py
│   ├── model_selection.py
│   ├── parameter_tuning.py
│   └── utilities.py
├── scripts/
│   ├── train_decision_tree.py
│   ├── train_linear_regression.py
│   ├── train_logistic_regression.py
│   ├── train_neural_network.py
│   ├── train_svm.py
│   └── compare_models.py
└── README.md
```

## Usage
1. Place your dataset in the `data/` directory as `dataset.csv`.
2. Run the training scripts:
```py
python train_linear_regression.py
```
```py
python train_logistic_regression.py
```

This AI Trainer project includes substantial code and can be extended further with additional models, more sophisticated preprocessing, hyperparameter tuning, and more.
