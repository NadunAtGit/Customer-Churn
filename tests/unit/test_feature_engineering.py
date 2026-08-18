import pandas as pd

from retention_ai.features.engineering import (
    add_engineered_features,
)


def test_engineered_features_created():

    df = pd.DataFrame({
        "Phone Service": ["Yes"],
        "Online Security": ["Yes"],
        "Online Backup": ["No"],
        "Device Protection": ["Yes"],
        "Tech Support": ["No"],
        "Streaming TV": ["Yes"],
        "Streaming Movies": ["No"],
        "Contract": ["Month-to-month"],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "Total Charges": [200.0],
        "Tenure Months": [4],
        "Monthly Charges": [50.0],
    })

    result = add_engineered_features(df)

    expected_columns = [
        "NumberOfServices",
        "IsMonthToMonth",
        "HasTechSupport",
        "HasOnlineSecurity",
        "HasPartnerOrDependents",
        "AvgMonthlySpend",
        "ChargesPerService",
    ]

    for col in expected_columns:
        assert col in result.columns

def test_engineered_feature_values():

    df = pd.DataFrame({
        "Phone Service": ["Yes"],
        "Online Security": ["Yes"],
        "Online Backup": ["No"],
        "Device Protection": ["Yes"],
        "Tech Support": ["No"],
        "Streaming TV": ["Yes"],
        "Streaming Movies": ["No"],
        "Contract": ["Month-to-month"],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "Total Charges": [200.0],
        "Tenure Months": [4],
        "Monthly Charges": [50.0],
    })

    result = add_engineered_features(df)

    assert result.loc[0, "NumberOfServices"] == 4
    assert result.loc[0, "IsMonthToMonth"] == 1
    assert result.loc[0, "HasTechSupport"] == 0
    assert result.loc[0, "HasOnlineSecurity"] == 1
    assert result.loc[0, "HasPartnerOrDependents"] == 1
    assert result.loc[0, "AvgMonthlySpend"] == 50.0

def test_zero_tenure_does_not_divide_by_zero():

    df = pd.DataFrame({
        "Phone Service": ["Yes"],
        "Online Security": ["No"],
        "Online Backup": ["No"],
        "Device Protection": ["No"],
        "Tech Support": ["No"],
        "Streaming TV": ["No"],
        "Streaming Movies": ["No"],
        "Contract": ["Month-to-month"],
        "Partner": ["No"],
        "Dependents": ["No"],
        "Total Charges": [0.0],
        "Tenure Months": [0],
        "Monthly Charges": [20.0],
    })

    result = add_engineered_features(df)

    assert result["AvgMonthlySpend"].notna().all()