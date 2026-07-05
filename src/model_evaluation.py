"""
model_evaluation.py
===================
Evaluation metrics, plots, and JSON report generation for regression
and classification models in the Voyage Analytics MLOps pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

from .utils import timer

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates machine learning models and saves performance reports/visualizations.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports")).resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelEvaluator initialised – reports_dir=%s", self.reports_dir)

    @timer
    def evaluate_regression(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "flight_price_regression",
    ) -> Dict[str, float]:
        """Evaluate a regression model and save metrics/charts.

        Parameters
        ----------
        model : Any
            Fitted regression model.
        X_test : pd.DataFrame
            Testing features.
        y_test : pd.Series
            Testing target.
        model_name : str
            Identifier for saving plots and reports.

        Returns
        -------
        dict[str, float]
            Evaluation metrics.
        """
        logger.info("Evaluating regression model '%s'...", model_name)
        preds = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, preds))
        mse = float(mean_squared_error(y_test, preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))
        mape = float(np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1.0))) * 100)

        metrics = {
            "mean_absolute_error": mae,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "r2_score": r2,
            "mean_absolute_percentage_error": mape,
        }
        logger.info("Regression Metrics for %s: %s", model_name, metrics)

        # Save metrics JSON
        self._save_json_report(metrics, f"{model_name}_metrics.json")

        # Save actual vs predicted plot
        try:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=preds, alpha=0.3, color="#00d4aa")
            # Diagonal line
            ideal_min = min(y_test.min(), preds.min())
            ideal_max = max(y_test.max(), preds.max())
            plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], color="#ff6b6b", linestyle="--")
            plt.xlabel("Actual Flight Price")
            plt.ylabel("Predicted Flight Price")
            plt.title(f"Actual vs Predicted Prices — {model_name}\n(R² = {r2:.4f})")
            plt.tight_layout()
            
            plot_path = self.reports_dir / f"{model_name}_actual_vs_predicted.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info("Saved actual vs predicted plot to %s", plot_path)
        except Exception as exc:
            logger.warning("Could not generate regression plots: %s", exc)

        # Feature Importance Plot
        self._save_feature_importances(model, X_test.columns.tolist(), model_name)

        return metrics

    @timer
    def evaluate_classification(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "classifier",
    ) -> Dict[str, Any]:
        """Evaluate a classification model and save metrics/charts.

        Parameters
        ----------
        model : Any
            Fitted classifier.
        X_test : pd.DataFrame
            Testing features.
        y_test : pd.Series
            Testing target.
        model_name : str
            Identifier for saving plots and reports.

        Returns
        -------
        dict[str, Any]
            Evaluation metrics.
        """
        logger.info("Evaluating classification model '%s'...", model_name)
        preds = model.predict(X_test)
        
        # Try getting probabilities
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)[:, 1]
            except Exception as exc:
                logger.warning("predict_proba failed: %s", exc)

        acc = float(accuracy_score(y_test, preds))
        
        # Precision, recall, f1 for binary classification (default class 1)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        
        auc_roc = 0.0
        if probs is not None:
            try:
                auc_roc = float(roc_auc_score(y_test, probs))
            except Exception as exc:
                logger.warning("roc_auc_score failed: %s", exc)

        metrics = {
            "accuracy": acc,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": auc_roc,
        }
        logger.info("Classification Metrics for %s: %s", model_name, metrics)

        # Save metrics JSON
        self._save_json_report(metrics, f"{model_name}_metrics.json")

        # Save confusion matrix plot
        try:
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Class 0", "Class 1"],
                        yticklabels=["Class 0", "Class 1"])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"Confusion Matrix — {model_name}\n(Accuracy = {acc:.4f})")
            plt.tight_layout()
            
            plot_path = self.reports_dir / f"{model_name}_confusion_matrix.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info("Saved confusion matrix plot to %s", plot_path)
        except Exception as exc:
            logger.warning("Could not generate classification plots: %s", exc)

        # Feature Importance Plot
        self._save_feature_importances(model, X_test.columns.tolist(), model_name)

        return metrics

    def _save_json_report(self, metrics: Dict[str, Any], filename: str) -> None:
        """Helper to write metrics dictionary to a JSON file."""
        report_path = self.reports_dir / filename
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            logger.info("Saved JSON metrics report to %s", report_path)
        except Exception as exc:
            logger.error("Failed to write JSON report to %s: %s", report_path, exc)

    def _save_feature_importances(self, model: Any, feature_names: List[str], model_name: str) -> None:
        """Extract and plot feature importances if the model supports it."""
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        if importances is None:
            logger.info("Model '%s' does not expose feature importances or coefficients.", model_name)
            return

        try:
            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False)

            # Save csv
            df_imp.to_csv(self.reports_dir / f"{model_name}_feature_importances.csv", index=False)

            # Plot horizontal bar chart
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=df_imp, palette="viridis")
            plt.xlabel("Importance Score")
            plt.ylabel("Feature")
            plt.title(f"Feature Importance — {model_name}")
            plt.tight_layout()
            
            plot_path = self.reports_dir / f"{model_name}_feature_importance.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info("Saved feature importance plot to %s", plot_path)
        except Exception as exc:
            logger.warning("Could not generate feature importance plots for '%s': %s", model_name, exc)
