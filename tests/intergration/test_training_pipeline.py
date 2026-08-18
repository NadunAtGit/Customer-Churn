from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
)

from retention_ai.data.ingestion import (
    load_data,
)

from retention_ai.data.preprocessing import (
    clean_data,
    split_features_target,
)

from retention_ai.features.engineering import (
    add_engineered_features,
)

from retention_ai.models.train import (
    build_model,
)

from retention_ai.models.evaluate import (
    evaluate_model,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco_customer_churn.xlsx"
)


def test_end_to_end_training_pipeline():

    df = load_data(DATA_PATH)

    # Small subset keeps the integration test fast
    df = df.sample(
        n=min(1000, len(df)),
        random_state=42,
    )

    df = clean_data(df)

    X, y = split_features_target(df)

    X = add_engineered_features(X)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42,
        )
    )

    model_params = {
        "iterations": 20,
        "depth": 4,
        "learning_rate": 0.1,
    }

    model = build_model(
        X_train,
        model_params,
    )

    model.fit(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=0.35,
    )

    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1