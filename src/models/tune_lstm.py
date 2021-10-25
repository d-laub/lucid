import argparse
from functools import partial

import optuna


# TODO: load data
def prep_data(path):

    return X_train, X_val, y_train, y_val

# 
def objective_lstm(trial: optuna.Trial, data):
    # suggest params
    params = {}

    # instantiate model
    # clf = LogisticRegression(**params)

    X_train, X_val, y_train, y_val = data

    # fit model
    # clf.fit(X_train, y_train)

    # return score
    # return clf.score(X_val, y_val)

    raise NotImplementedError

def main(args):
    # TODO: set seed
    # np.random.seed(args.seed)

    data = prep_data(args.data_path)

    objective = partial(objective_lstm, data=data)

    study = optuna.create_study(study_name=f'tune_{args.model}')
    study.optimize(objective, n_trials=args.n_trials)

if __name__ == '__main__':
    raise NotImplementedError
    parser = argparse.ArgumentParser(description='Tune an LSTM with Optuna.')
    parser.add_argument('data_path', help='Path to training data.')
    parser.add_argument('model', choices=[], help='Model type.')
    parser.add_argument('--model_path', help='Path to model weights.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for tuning.')
    args = parser.parse_args()

    main(args)