import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates
from deepchecks.tabular.checks import ConflictingLabels


def test_data():
    # Load dataset
    df = pd.read_csv("data/spam.csv")
    dataset = Dataset(
        df, cat_features=["text"], label="label", features=["text"]
    )
    # References:
    # https://docs.deepchecks.com/stable/checks_gallery/tabular/data_integrity/plot_data_duplicates.html
    # https://docs.deepchecks.com/stable/checks_gallery/tabular/data_integrity/plot_conflicting_labels.html
    # Run the checks
    duplicates = DataDuplicates().run(dataset).display
    conflicting_labels = ConflictingLabels().run(dataset).display
    assert len(duplicates) == 0
    assert len(conflicting_labels) == 0
