import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics


def test_train_model():
    """Test that train_model returns a trained model"""
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_inference():
    """Test that inference returns predictions"""
    X_train = np.random.rand(50, 5)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.rand(10, 5)

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert len(preds) == 10
    assert all(p in [0, 1] for p in preds)


def test_compute_model_metrics():
    """Test that metrics are calculated correctly"""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
