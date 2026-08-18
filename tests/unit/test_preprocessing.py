import pandas as pd

from retention_ai.data.preprocessing import (
    clean_data,
)


def test_total_charges_converted_to_numeric():

    df = pd.DataFrame({
        "Total Charges": [
            "100.50",
            "200.00",
            " "
        ]
    })

    result = clean_data(df)

    assert pd.api.types.is_numeric_dtype(
        result["Total Charges"]
    )

    assert result["Total Charges"].isna().sum() == 1

from retention_ai.data.preprocessing import (
    split_features_target,
)


def test_target_removed_from_features():

    df = pd.DataFrame({
        "CustomerID": ["A1"],
        "Count": [1],
        "Country": ["United States"],
        "State": ["California"],
        "City": ["Los Angeles"],
        "Zip Code": [90001],
        "Lat Long": ["x"],
        "Latitude": [1.0],
        "Longitude": [2.0],
        "Churn Label": ["Yes"],
        "Churn Value": [1],
        "Churn Score": [90],
        "Churn Reason": ["Price"],
        "Tenure Months": [5],
    })

    X, y = split_features_target(df)

    assert "Churn Value" not in X.columns
    assert "Churn Label" not in X.columns
    assert "Churn Score" not in X.columns
    assert "Churn Reason" not in X.columns

    assert y.iloc[0] == 1

from retention_ai.data.preprocessing import (
    build_preprocessor,
)


def test_preprocessor_transforms_data():

    X = pd.DataFrame({
        "Tenure Months": [2, 10, 20],
        "Monthly Charges": [30.0, 60.0, 90.0],
        "Gender": [
            "Male",
            "Female",
            "Male"
        ],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year"
        ],
    })

    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > X.shape[1]

def test_preprocessor_handles_unknown_category():

    train = pd.DataFrame({
        "Tenure Months": [1, 10],
        "Gender": [
            "Male",
            "Female"
        ]
    })

    test = pd.DataFrame({
        "Tenure Months": [5],
        "Gender": [
            "Unknown"
        ]
    })

    preprocessor = build_preprocessor(train)

    preprocessor.fit(train)

    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1