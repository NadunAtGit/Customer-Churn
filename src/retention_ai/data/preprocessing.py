import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


TARGET = "Churn Value"

DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",

    # Leakage
    "Churn Label",
    "Churn Score",
    "Churn Reason",
]


def clean_data(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce",
    )

    return df


def split_features_target(
    df: pd.DataFrame
):
    y = df[TARGET]

    X = df.drop(
        columns=DROP_COLUMNS + [TARGET],
        errors="ignore",
    )

    return X, y


def build_preprocessor(
    X: pd.DataFrame
) -> ColumnTransformer:

    numeric_features = (
        X.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )