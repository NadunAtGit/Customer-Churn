from pydantic import BaseModel


class CustomerInput(BaseModel):
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Tenure_Months: int

    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str

    Contract: str
    Paperless_Billing: str
    Payment_Method: str

    Monthly_Charges: float
    Total_Charges: float
    CLTV: int


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str
    risk_level: str
    threshold: float