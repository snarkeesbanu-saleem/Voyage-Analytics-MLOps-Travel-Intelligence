"""
tracking_config.py
===================
MLflow tracking setup and experiment logging utilities for Voyage Analytics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Manages tracking URI, experiments, active runs, and metric logging in MLflow.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing tracking preferences.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        
        # Setup local sqlite-based tracking by default to prevent FileStore maintenance errors
        project_root = Path(__file__).resolve().parent.parent
        default_tracking_uri = f"sqlite:///{project_root.as_posix()}/mlflow.db"
        
        self.tracking_uri = self.config.get("mlflow", {}).get("tracking_uri", default_tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info("MLflow Tracking URI set to: %s", self.tracking_uri)

    def setup_experiment(self, experiment_name: str) -> str:
        """Create or retrieve an experiment by name and set it active.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.

        Returns
        -------
        str
            The experiment ID.
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                exp_id = mlflow.create_experiment(experiment_name)
                logger.info("Created new MLflow experiment: %s (ID: %s)", experiment_name, exp_id)
            else:
                exp_id = experiment.experiment_id
                logger.info("Found existing MLflow experiment: %s (ID: %s)", experiment_name, exp_id)
            
            mlflow.set_experiment(experiment_name)
            return exp_id
        except Exception as exc:
            logger.error("Failed to setup MLflow experiment '%s': %s", experiment_name, exc)
            raise

    def log_run(
        self,
        experiment_name: str,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model: Any,
        model_artifact_name: str,
        artifacts: Optional[list[str]] = None,
    ) -> None:
        """Log parameters, metrics, model, and artifacts under a named run.

        Parameters
        ----------
        experiment_name : str
            Name of the active experiment.
        run_name : str
            Name of this specific run.
        params : dict
            Hyperparameters or execution configs to log.
        metrics : dict
            Evaluation metrics to log.
        model : Any
            The fitted sklearn model to log as an MLflow artifact.
        model_artifact_name : str
            Folder name to save the logged model.
        artifacts : list[str] | None
            List of filepaths to upload as run artifacts.
        """
        self.setup_experiment(experiment_name)
        
        logger.info("Starting MLflow run '%s' under experiment '%s'...", run_name, experiment_name)
        try:
            with mlflow.start_run(run_name=run_name):
                # Log configs/params
                mlflow.log_params(params)
                logger.debug("Logged parameters: %s", params)

                # Log metrics
                mlflow.log_metrics(metrics)
                logger.debug("Logged metrics: %s", metrics)

                # Log sklearn model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_artifact_name,
                    registered_model_name=model_artifact_name
                )
                logger.info("Logged and registered model artifact: %s", model_artifact_name)

                # Log extra artifacts
                if artifacts:
                    for artifact_path in artifacts:
                        path_obj = Path(artifact_path)
                        if path_obj.exists():
                            mlflow.log_artifact(str(path_obj))
                            logger.info("Logged extra artifact: %s", path_obj.name)
                        else:
                            logger.warning("Artifact not found: %s", artifact_path)
        except Exception as exc:
            logger.error("Failed to log MLflow run '%s': %s", run_name, exc)
            # Re-raise so execution pipelines are aware
            raise
