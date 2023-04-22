import pandas as pd
import pytest
# src
from src.data.make_preprocessing import drop_exact_duplicates


@pytest.fixture
def data():
    data_test = pd.read_csv('data/raw/Original_Dataset.csv')
    return data_test


def test_drop_exact_duplicates(data):
    # Given
    len_before = len(data)
    # When
    data_scenario = drop_exact_duplicates(data)
    len_after = len(data_scenario)

    # Then
    assert len_before is len_after
