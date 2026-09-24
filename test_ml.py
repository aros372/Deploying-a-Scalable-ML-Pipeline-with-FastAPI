import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import compute_model_metrics

def test_binary_encoding():
    # check the process_data function is binary encoding the label properly
    df = pd.DataFrame({'feat_1':[1,2,3], 'feat_2':[4,5,6], 'label':['a','b','a']})
    _, y, _, _ =process_data(df, ['feat_1', 'feat_2'], 'label', True)
    assert set(y) == {0,1}

def test_feature_transformation():
    # check the process_data function not dropping any rows from the feature or label sets
    df = pd.DataFrame({'feat_1':[1,2,3], 'feat_2':[4,5,6], 'label':['a','b','a']})
    X, y, _, _ =process_data(df, ['feat_1', 'feat_2'], 'label', True)
    assert df.shape[0] == X.shape[0] and df.shape[0] == y.shape[0]

def test_model_metrics():
    y_test = np.array([0,0,1,1])
    preds = np.array([0,1,0,1])
    p, r, fb = compute_model_metrics(y_test, preds)
    assert all(val==0.5 for val in [p, r, fb])