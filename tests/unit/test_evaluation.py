import numpy as np

from retention_ai.models.evaluate import (
    evaluate_model,
)


class FakeModel:

    def predict_proba(self, X):
        return np.array([
            [0.8, 0.2],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.9, 0.1],
        ])


def test_evaluation_returns_metrics():

    model = FakeModel()

    X = np.zeros((4, 2))

    y = np.array([
        0,
        1,
        1,
        0,
    ])

    metrics = evaluate_model(
        model,
        X,
        y,
        threshold=0.35,
    )

    expected_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    for metric in expected_metrics:
        assert metric in metrics