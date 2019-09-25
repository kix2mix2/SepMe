import os
from dotenv import load_dotenv

load_dotenv(
    verbose=True, override=False
)  # won't override env vars already set

config = {
    "LOG_FILE": os.getenv("LOG_FILE", "../sepme.log"),
    "MLFLOW_TRACKING_URI": os.getenv(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    ),
}

# TODO: env variable validation
