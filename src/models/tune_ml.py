import argparse
from functools import partial
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna

# TODO: load data
def prep_data(path):

    return X_train, X_val, y_train, y_val

def objective_LR(trial: optuna.Trial, data):
    """Objective for logistic regression."""

    params = {
        'penalty': 'elasticnet',
        'solver': 'saga', # req feature standardization
        'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1)
    }

    clf = LogisticRegression(**params)

    X_train, X_val, y_train, y_val = data

    clf.fit(X_train, y_train)

    return clf.score(X_val, y_val)

def objective_SVM(trial: optuna.Trial, data):
    """Objective for support vector machine."""

    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])

    if kernel == 'linear':
        params = {
            'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
        }
        clf = LinearSVC(**params)
    elif kernel == 'rbf':
        params = {
            'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 1e3, log=True)
        }
        clf = SVC(**params)

    X_train, X_val, y_train, y_val = data

    clf.fit(X_train, y_train)

    return clf.score(X_val, y_val)

def objective_RF(trial: optuna.Trial, data):
    """Objective for random forest."""

    params = {
        # TODO: suggest hyperparams
    }

    clf = RandomForestClassifier(**params)

    X_train, X_val, y_train, y_val = data

    clf.fit(X_train, y_train)

    return clf.score(X_val, y_val)

def objective_XGB(trial: optuna.Trial, data):
    """Objective for XGBoost."""

    params = {
        'objective': 'reg:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1),
        'subsample': trial.suggest_float('subsample', 0.6, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }

    clf = XGBClassifier(**params)

    X_train, X_val, y_train, y_val = data

    clf.fit(X_train, y_train)

    return clf.score(X_val, y_val)

def main(args):
    np.random.seed(args.seed)

    data = prep_data(args.data_path)

    objectives = {'RF': objective_RF, 'XGB': objective_XGB}
    objective = partial(objectives[args.model], data=data)

    study = optuna.create_study(study_name=f'tune_{args.model}')
    study.optimize(objective, n_trials=args.n_trials)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune classic ML algorithms with Optuna.')
    parser.add_argument('data_path', help='Path to training data.')
    parser.add_argument('model', choices=['RF', 'XGB'], help='Model type.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for tuning.')
    args = parser.parse_args()

    main(args)