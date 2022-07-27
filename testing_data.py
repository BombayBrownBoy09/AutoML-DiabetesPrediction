import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

def test_data_integrity():
    # The label can be passed as a column name or a separate pd.Series / pd.DataFrame
    df = pd.read_csv("data/diabetes.csv")
    ds = Dataset(df, label="Outcome")
    # Run Suite
    integ_suite = data_integrity().remove(0)
    suite_result = integ_suite.run(ds)
    suite_result.save_as_html(file="deepchecks.html")
    assert len(suite_result.get_not_passed_checks()) == 0
