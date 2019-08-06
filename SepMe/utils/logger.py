import logging
import mlflow

class MLFlowLogger(logging.getLoggerClass()):
    """Logger class that provides simple wrapper functions to simultaneously log things
    using mlflow as well as normal logging handlers.
    """

    def log_param(self, key, value, level=logging.INFO):
        mlflow.log_param(key, value)
        self.log(level, f"Param logged: {key}={value}")

    def log_metric(self, key, value, level=logging.INFO):
        mlflow.log_metric(key, value)
        self.log(level, f"Metric logged: {key}={value}")

    def log_artifact(self, local_path, artifact_path=None, level=logging.INFO):
        mlflow.log_artifact(local_path, artifact_path)
        self.log(level, f"Artifact logged: {local_path} to {artifact_path or 'default location'}")

    def log_artifacts(self, local_dir, artifact_path=None, level=logging.INFO):
        mlflow.log_artifacts(local_dir, artifact_path)
        self.log(level, f"Artifacts logged: {local_dir} to {artifact_path or 'default location'}")

    def set_tag(self, key, value, level=logging.INFO):
        mlflow.set_tag(key, value)
        self.log(level, f"Tag set: {key}: {value}")


def get_logger(name, log_file):
    """Create a custom logger that sends all messages (inc. debug) to `log_file` and
    info and above to console.

    :param name: The name of the logger passed to Logger object
    :type name: string
    :param log_file: The file to send log messages to
    :type log_file: string
    :return: The logger object
    :rtype: Logging.Logger
    """
    logger = MLFlowLogger(name)
    logger.setLevel(logging.INFO)

    debug_handler = logging.FileHandler(log_file)
    debug_handler.setLevel(logging.INFO)
    debug_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    debug_handler.setFormatter(debug_format)

    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    info_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    info_handler.setFormatter(info_format)

    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)

    return logger
