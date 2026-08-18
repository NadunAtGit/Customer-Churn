import mlflow
import mlflow.sklearn


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str = "http://127.0.0.1:5000",
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_training_run(
    run_name: str,
    model,
    params: dict,
    metrics: dict,
    threshold: float,
    input_example=None,
):
    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.log_params(params)

        mlflow.log_param(
            "decision_threshold",
            threshold,
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            skops_trusted_types=[
                "catboost.core.CatBoostClassifier",
                "numpy.dtype",
            ],
        )