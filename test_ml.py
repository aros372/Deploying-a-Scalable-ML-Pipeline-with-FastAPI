import pytest
import pandas as pd
from ml.data import process_data
from train_model import train_test_split, train_model, inference,compute_model_metrics

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

def test_metric_ranges():
    df = pd.DataFrame({'feat_1':[1,2,3,4,1,2,3,4], 'feat_2':[9,10,11,12,12,11,10,9], 'label':['a','b','a','a','a','b','a','b',]})
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train, encoder, lb = process_data(train, ['feat_1', 'feat_2'], 'label', True)
    X_test, y_test, _, _ = process_data(test, ['feat_1', 'feat_2'], 'label', False, encoder, lb)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    p, r, fb = compute_model_metrics(y_test, preds)
    assert all(0<=val<=1 for val in [p, r, fb])

