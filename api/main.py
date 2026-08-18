from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import (
    CustomerInput,
    PredictionResponse,
)

from api.model_service import (
    ChurnModelService,
)


model_service = ChurnModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading churn model...")

    model_service.load_model()

    print("Model loaded successfully")

    yield

    print("Application shutting down")


app = FastAPI(
    title="RetentionAI API",
    description=(
        "Customer churn prediction and "
        "retention intelligence API"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message":
            "RetentionAI API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded":
            model_service.model is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    customer: CustomerInput
):

    try:

        customer_dict = (
            customer.model_dump()
        )

        result = (
            model_service.predict(
                customer_dict
            )
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc