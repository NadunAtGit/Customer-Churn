import pandas as pd


def add_engineered_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    service_cols = [
        "Phone Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]

    df["NumberOfServices"] = (
        df[service_cols]
        .eq("Yes")
        .sum(axis=1)
    )

    df["IsMonthToMonth"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    df["HasTechSupport"] = (
        df["Tech Support"] == "Yes"
    ).astype(int)

    df["HasOnlineSecurity"] = (
        df["Online Security"] == "Yes"
    ).astype(int)

    df["HasPartnerOrDependents"] = (
        (df["Partner"] == "Yes")
        | (df["Dependents"] == "Yes")
    ).astype(int)

    df["AvgMonthlySpend"] = (
        df["Total Charges"]
        / df["Tenure Months"].replace(0, 1)
    )

    df["ChargesPerService"] = (
        df["Monthly Charges"]
        / df["NumberOfServices"].replace(0, 1)
    )

    return df