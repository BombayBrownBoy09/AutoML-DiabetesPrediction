import pprint
import pandas as pd
from ludwig.automl import auto_train

# Reading the csv file and storing it in a dataframe.
inputdf = pd.read_csv("data/diabetes.csv")

auto_train_results = auto_train(
    dataset=inputdf, target="Outcome", time_limit_s=120, tune_for_memory=True
)

pprint.pprint(auto_train_results)
