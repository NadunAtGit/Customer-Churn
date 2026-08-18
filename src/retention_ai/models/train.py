from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline

from retention_ai.data.preprocessing import (
    build_preprocessor,
)


FINAL_THRESHOLD = 0.35


def build_model(
    X_train,
    model_params: dict,
) -> Pipeline:

    preprocessor = build_preprocessor(
        X_train
    )

    classifier = CatBoostClassifier(
        **model_params,
        random_state=42,
        verbose=0,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return pipeline