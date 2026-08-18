import mlflow
import pandas as pd

from retention_ai.features.engineering import (
    add_engineered_features,
)

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
MODEL_URI = "models:/CustomerRetentionModel@champion"

THRESHOLD = 0.35


class ChurnModelService:

    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = mlflow.sklearn.load_model(
            MODEL_URI
        )

    def predict(self, customer_data: dict):

        if self.model is None:
            raise RuntimeError(
                "Model has not been loaded"
            )

        row = {
            "Gender":
                customer_data["Gender"],

            "Senior Citizen":
                customer_data["Senior_Citizen"],

            "Partner":
                customer_data["Partner"],

            "Dependents":
                customer_data["Dependents"],

            "Tenure Months":
                customer_data["Tenure_Months"],

            "Phone Service":
                customer_data["Phone_Service"],

            "Multiple Lines":
                customer_data["Multiple_Lines"],

            "Internet Service":
                customer_data["Internet_Service"],

            "Online Security":
                customer_data["Online_Security"],

            "Online Backup":
                customer_data["Online_Backup"],

            "Device Protection":
                customer_data["Device_Protection"],

            "Tech Support":
                customer_data["Tech_Support"],

            "Streaming TV":
                customer_data["Streaming_TV"],

            "Streaming Movies":
                customer_data["Streaming_Movies"],

            "Contract":
                customer_data["Contract"],

            "Paperless Billing":
                customer_data["Paperless_Billing"],

            "Payment Method":
                customer_data["Payment_Method"],

            "Monthly Charges":
                customer_data["Monthly_Charges"],

            "Total Charges":
                customer_data["Total_Charges"],

            "CLTV":
                customer_data["CLTV"],
        }

        df = pd.DataFrame(
            [row]
        )

        # Your current MLflow model expects
        # engineered features.
        df = add_engineered_features(df)

        probability = (
            self.model
            .predict_proba(df)[:, 1][0]
        )

        prediction = (
            "Churn"
            if probability >= THRESHOLD
            else "Retained"
        )

        if probability >= 0.70:
            risk = "High"
        elif probability >= THRESHOLD:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "churn_probability":
                round(float(probability), 4),

            "prediction":
                prediction,

            "risk_level":
                risk,

            "threshold":
                THRESHOLD,
        }