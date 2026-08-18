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
    FINAL_THRESHOLD,
)

from retention_ai.models.evaluate import (
    evaluate_model,
)

from retention_ai.tracking.mlflow_tracker import (
    setup_mlflow,
    log_training_run,
)


DATA_PATH = Path(
    "../data/raw/Telco_customer_churn.xlsx"
)


def main():

    # -------------------------
    # Load
    # -------------------------

    df = load_data(
        DATA_PATH
    )

    # -------------------------
    # Clean
    # -------------------------

    df = clean_data(
        df
    )

    # -------------------------
    # Split X/y
    # -------------------------

    X, y = split_features_target(
        df
    )

    # -------------------------
    # Feature engineering
    # -------------------------

    X = add_engineered_features(
        X
    )

    # -------------------------
    # Train/test split
    # -------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    # -------------------------
    # Tuned CatBoost params
    # REPLACE with your actual
    # tuning result
    # -------------------------

    model_params = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
    }

    # -------------------------
    # Build + train
    # -------------------------

    model = build_model(
        X_train,
        model_params,
    )

    model.fit(
        X_train,
        y_train,
    )

    # -------------------------
    # Evaluate
    # -------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=FINAL_THRESHOLD,
    )

    print("\nFinal Metrics")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    # -------------------------
    # MLflow
    # -------------------------

    setup_mlflow(
        experiment_name=(
            "RetentionAI-Churn"
        )
    )

    input_example = (
        X_train.head(5)
    )

    log_training_run(
        run_name=(
            "catboost-production"
        ),
        model=model,
        params=model_params,
        metrics=metrics,
        threshold=FINAL_THRESHOLD,
        input_example=input_example,
    )


if __name__ == "__main__":
    main()