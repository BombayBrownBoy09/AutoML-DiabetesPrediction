import sklearn.metrics
import pandas as pd
from flaml import AutoML

input_df = pd.read_csv("data/neo.csv")
X = input_df.drop(["Outcome"], axis=1)
y = input_df["Outcome"]  # pylint: disable=E1136

automl = AutoML()

X_train, y_train = X,y

settings = {
    "time_budget": 50,  # total running time in seconds
    "metric": "accuracy", 
    "task": "classification",  # task type
    "seed": 7654321
}

automl.fit(X_train=X_train, y_train=y_train, **settings)

# '''retrieve best config and best learner'''
print('Best hyperparmeter config:', automl.best_config)
print('Best accuracy on validation data: {0:.4g}'.format(1-automl.best_loss))
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
