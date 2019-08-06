import mlflow
from SepMe.utils.logger import get_logger
from SepMe.utils.config import config

logger = get_logger("SepMe", config["LOG_FILE"])
logger.info(f"MLFLOW_TRACKING_URI - {config['MLFLOW_TRACKING_URI']}")

mlflow.set_tracking_uri(config["MLFLOW_TRACKING_URI"])
