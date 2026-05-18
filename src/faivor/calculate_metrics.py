import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix

from faivor.model_metadata import ModelMetadata
from faivor.parse_data import detect_delimiter, detect_decimal
from faivor.metrics.regression import metrics as regression_metrics
from faivor.metrics.classification import metrics as classification_metrics
from faivor.utils import convert_to_json_serializable, safe_divide

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Calculate metrics for model predictions and perform subgroup analysis.
    """
    
    def __init__(self, model_metadata: ModelMetadata, predictions: List, 
                expected_outputs: List, inputs: Optional[List] = None):
        """
        Initialize the metrics calculator.
        
        Parameters
        ----------
        model_metadata : ModelMetadata
            Model metadata containing information about the model.
        predictions : List
            Predictions from the model.
        expected_outputs : List
            Expected outputs (ground truth) for evaluation.
        inputs : List, optional
            Input data used for making predictions, by default None.
        """
        self.model_metadata = model_metadata
        self.predictions = predictions
        self.expected_outputs = expected_outputs
        self.inputs = inputs
        self.output_field = model_metadata.output
        self.metadata_json = model_metadata.metadata
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare data for metrics calculation by filtering valid samples.
        
        Returns
        -------
        tuple
            Tuple containing (y_true, y_pred, valid_indices)
            
        Raises
        ------
        ValueError
            If predictions and expected outputs have different lengths or are empty
        """
        if not self.predictions or not self.expected_outputs:
            raise ValueError("Predictions and expected outputs cannot be empty")
            
        if len(self.predictions) != len(self.expected_outputs):
            raise ValueError(
                f"Predictions (length={len(self.predictions)}) and expected outputs "
                f"(length={len(self.expected_outputs)}) must have the same length"
            )
        
        valid_indices = []
        y_true_values = []
        y_pred_values = []
        
        for i, (output, pred) in enumerate(zip(self.expected_outputs, self.predictions)):
            try:
                # need to check if expected output has the required key and is numeric
                if isinstance(output, dict) and self.output_field in output:
                    true_val = output[self.output_field]
                    if true_val not in ('x', 'X', ''):
                        true_float = float(true_val)
                        pred_float = float(pred)

                        if not np.isfinite(true_float) or not np.isfinite(pred_float):
                            continue
                        
                        valid_indices.append(i)
                        y_true_values.append(true_float)
                        y_pred_values.append(pred_float)
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping sample {i} due to conversion error: {e}")
                continue
        
        if not valid_indices:
            raise ValueError(
                f"No valid numeric data found. Please check that the output field "
                f"'{self.output_field}' contains numeric values."
            )
        
        y_true = np.array(y_true_values)
        y_pred = np.array(y_pred_values)
        
        return y_true, y_pred, valid_indices
    
    def get_sensitive_values(self, valid_indices: List[int]) -> Optional[np.ndarray]:
        """
        Get sensitive attribute values for valid indices if available.
        
        Parameters
        ----------
        valid_indices : List[int]
            Indices of valid data points.
            
        Returns
        -------
        np.ndarray or None
            Array of sensitive values if available, None otherwise.
        """
        sensitive_attribute = self.metadata_json.get("sensitive_attribute")
        if not sensitive_attribute or not self.inputs:
            return None
            
        sensitive_values = []
        for i in valid_indices:
            if isinstance(self.inputs[i], dict) and sensitive_attribute in self.inputs[i]:
                sensitive_values.append(self.inputs[i][sensitive_attribute])
            else:
                sensitive_values.append(None)
        
        if any(v is not None for v in sensitive_values):
            return np.array(sensitive_values)
        return None
    
    def get_categorical_features_from_json(self, json_path: Path) -> List[str]:
        """
        Read a JSON file that describes column metadata and extract categorical column names.
        
        Parameters
        ----------
        json_path : Path
            Path to the JSON file with column metadata.
            
        Returns
        -------
        List[str]
            List of categorical column names.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            column_metadata = json.load(f)
        
        categorical_columns = []
        for column in column_metadata.get("columns", []):
            if column.get("categorical", False):
                    col_name = column.get("name_model", column.get("name_csv"))
                    categorical_columns.append(col_name)
        return categorical_columns
    
    def _compute_metrics_for_category(self, metrics_collection, category_name, y_true, y_pred, 
                                    all_metrics, sensitive_values=None, feature_importance=None):
        """
        Helper method to compute metrics for a specific category.
        
        Parameters
        ----------
        metrics_collection : list
            Collection of metric objects to compute
        category_name : str
            Name of metric category (performance, fairness, explainability)
        y_true : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Predicted values
        all_metrics : dict
            Dictionary to store results
        sensitive_values : np.ndarray, optional
            Sensitive attribute values for fairness metrics
        feature_importance : dict, optional
            Feature importance values for explainability metrics
        """
        for metric in metrics_collection:
            try:
                # special case for demographic parity ratio (fairness)
                if metric.function_name == "demographic_parity_ratio":
                    if sensitive_values is not None:
                        result = metric.compute(y_true, y_pred, sensitive_values)
                        all_metrics[f"{category_name}.{metric.regular_name}"] = convert_to_json_serializable(result)
                    else:
                        all_metrics[f"{category_name}.{metric.regular_name}"] = "Requires sensitive attributes (not computed)"
                    continue
                    
                # special case for feature importance ratio (explainability)
                if metric.function_name == "feature_importance_ratio":
                    if feature_importance:
                        result = metric.compute(y_true, y_pred, feature_importance=feature_importance)
                        all_metrics[f"{category_name}.{metric.regular_name}"] = convert_to_json_serializable(result)
                    else:
                        all_metrics[f"{category_name}.{metric.regular_name}"] = "Requires feature importances (not computed)"
                    continue
                    
                # general case, compute metric with just ytrue and ypred
                result = metric.compute(y_true, y_pred)
                all_metrics[f"{category_name}.{metric.regular_name}"] = convert_to_json_serializable(result)
            except Exception as e:
                all_metrics[f"{category_name}.{metric.regular_name}"] = f"Error: {str(e)}"    
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics for the model.
        
        Returns
        -------
        dict
            Dictionary containing calculated metrics.
            
        Raises
        ------
        ValueError
            If data preparation fails or no valid data is found
        RuntimeError
            If metric calculation fails
        """
        try:
            y_true, y_pred, valid_indices = self.prepare_data()
        except ValueError as e:
            raise ValueError(f"Data preparation failed: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during data preparation: {str(e)}. "
                "Please check your data format and try again."
            ) from e
        
        if len(y_true) == 0:
            raise ValueError(
                "No valid numeric data pairs found for metric calculation. "
                "Please ensure your data contains numeric values for both predictions and expected outputs."
            )
            
        model_type = self.metadata_json.get("model_type", "regression")
        
        # sensitive attribute
        try:
            sensitive_values = self.get_sensitive_values(valid_indices)
        except Exception as e:
            logger.warning(f"Failed to get sensitive values: {e}")
            sensitive_values = None
            
        feature_importance = self.metadata_json.get("feature_importance")
        
        all_metrics = {}
        
        # detect appropriate metrics module based on model type
        if model_type.lower() not in ["classification", "regression"]:
            logger.warning(f"Unknown model type '{model_type}', defaulting to regression metrics")
            
        metrics_module = (classification_metrics if model_type.lower() == "classification" 
                        else regression_metrics)
        
        # calc metrics for each category
        try:
            self._compute_metrics_for_category(
                metrics_module.performance, "performance", 
                y_true, y_pred, all_metrics
            )
        except Exception as e:
            logger.error(f"Failed to compute performance metrics: {e}")
            all_metrics["performance.error"] = f"Failed to compute performance metrics: {str(e)}"
        
        try:
            self._compute_metrics_for_category(
                metrics_module.fairness, "fairness", 
                y_true, y_pred, all_metrics, 
                sensitive_values=sensitive_values
            )
        except Exception as e:
            logger.error(f"Failed to compute fairness metrics: {e}")
            all_metrics["fairness.error"] = f"Failed to compute fairness metrics: {str(e)}"
        
        try:
            self._compute_metrics_for_category(
                metrics_module.explainability, "explainability", 
                y_true, y_pred, all_metrics, 
                feature_importance=feature_importance
            )
        except Exception as e:
            logger.error(f"Failed to compute explainability metrics: {e}")
            all_metrics["explainability.error"] = f"Failed to compute explainability metrics: {str(e)}"
        
        if not all_metrics:
            raise RuntimeError(
                "Failed to calculate any metrics. Please check your model outputs and data format."
            )
                
        return all_metrics    

    def preprocess_probability_values(self, y_pred: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, str]:
        """
        Preprocess model outputs to ensure they are valid probabilities in [0,1] range.
        
        Parameters
        ----------
        y_pred : np.ndarray
            The prediction values to process
        verbose : bool, optional
            Whether to print information about the transformation, by default False
            
        Returns
        -------
        Tuple[np.ndarray, str]
            The preprocessed probabilities and a message describing the transformation
        """
        min_val = np.min(y_pred)
        max_val = np.max(y_pred)
        
        # check if values are already in valid prob ([0,1]) range
        if min_val >= 0 and max_val <= 1:
            return y_pred, "No transformation needed"
        
        #### treatment type #1 1: If values look like logits (large range, both positive and negative)
        if min_val < -1 or max_val > 2:
            if verbose:
                logger.info(f"Values range [{min_val:.4f}, {max_val:.4f}] looks like logits, applying sigmoid")
            transformed = 1 / (1 + np.exp(-y_pred))
            return transformed, "Applied sigmoid transformation (logits to probabilities)"
        
        ### treatment type #2: If values are in a consistent range but shifted/scaled
        if (max_val - min_val) > 0:
            if verbose:
                logger.info(f"Values in range [{min_val:.4f}, {max_val:.4f}], applying min-max scaling")
            transformed = (y_pred - min_val) / (max_val - min_val)
            return transformed, "Applied min-max scaling to [0,1] range"
        
        ### treatment type #3: Fallback to clipping
        if verbose:
            logger.info(f"Values outside [0,1] range, applying clipping")
        transformed = np.clip(y_pred, 0, 1)
        return transformed, "Applied clipping to [0,1] range"    
    
    def prepare_probability_data(self) -> Tuple[np.ndarray, np.ndarray, List[int], Optional[str]]:
        """
        Prepare binary classification data for threshold-based metrics calculation.
        
        Returns
        -------
        tuple
            Tuple containing (y_true, y_prob, valid_indices, transformation_message)
            where y_true contains binary labels (0 or 1),
            y_prob contains probability predictions for the positive class,
            valid_indices contains the indices of valid data points,
            and transformation_message describes any applied transformations.
        """
        # get the regular data
        y_true, y_pred, valid_indices = self.prepare_data()
        
        # check whether we have binary classification data (0 or 1)
        binary_mask = np.isin(y_true, [0, 1])
        if not np.all(binary_mask):
            # filter to keep only binary data
            y_true = y_true[binary_mask]
            y_pred = y_pred[binary_mask]
            valid_indices = [valid_indices[i] for i, is_binary in enumerate(binary_mask) if is_binary]
        
        # for probability, we'll use the predicted values directly and check if they need preprocessing
        y_prob, transformation_message = self.preprocess_probability_values(y_pred, verbose=True)
        
        return y_true, y_prob, valid_indices, transformation_message
    
    def calculate_threshold_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics across different classification thresholds for binary classification.
        
        Based on https://www.mdpi.com/2072-4292/13/13/2450.
        
        This method analyzes how different probability thresholds affect classification metrics,
        including precision, recall, F1-score, and confusion matrix components. It also calculates
        ROC curve and Precision-Recall curve data.
        
        Returns
        -------
        dict
            Dictionary containing threshold analysis results including:
            - ROC curve data (FPR, TPR, thresholds, AUC)
            - Precision-Recall curve data (precision, recall, thresholds, average precision)
            - Metrics at different thresholds (confusion matrix, precision, recall, F1, etc.)
        """
        
        # get binary ground truth and probability predictions
        y_true, y_prob, valid_indices, transformation_message = self.prepare_probability_data()
        
        # check for the edge case of zero binary data
        if len(y_true) == 0:
            return {"error": "No valid binary data pairs found for threshold analysis"}
        
        # double ckeck whether we have binary class labels (0 or 1)
        if not np.all(np.isin(y_true, [0, 1])):
            return {"error": "Threshold analysis requires binary ground truth labels (0 or 1)"}
        
        # calculate ROC curve
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            
            # make serializable
            roc_curve_data = {
                "fpr": convert_to_json_serializable(fpr),
                "tpr": convert_to_json_serializable(tpr),
                "thresholds": convert_to_json_serializable(roc_thresholds),
                "auc": float(auc_score)
            }
        except Exception as e:
            roc_curve_data = {"error": f"Could not compute ROC curve: {str(e)}"}
        
        # calculate Precision-Recall curve
        try:
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            
            # make serializable
            pr_curve_data = {
                "precision": convert_to_json_serializable(precision),
                "recall": convert_to_json_serializable(recall),
                "thresholds": convert_to_json_serializable(pr_thresholds),
                "average_precision": float(avg_precision)
            }
        except Exception as e:
            pr_curve_data = {"error": f"Could not compute PR curve: {str(e)}"}
        
        # calculate metrics for different thresholds, we'll go with the extreme case of 101 thresholds from 0.00 to 1.00
        thresholds = np.linspace(0, 1, 101)
        threshold_metrics = {}
        
        for threshold in thresholds:
            # create binary predictions using this threshold
            y_pred_binary = (y_prob >= threshold).astype(int)
            
            # calculate confusion matrix
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
                
                # calculate derived metrics
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = safe_divide(tp, tp + fp)
                recall = safe_divide(tp, tp + fn)
                specificity = safe_divide(tn, tn + fp)
                f1_score = safe_divide(2 * precision * recall, precision + recall)
                fpr = safe_divide(fp, fp + tn)
                
                # stash metrics
                threshold_metrics[str(threshold)] = {
                    "confusion_matrix": {
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp)
                    },
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "specificity": float(specificity),
                    "f1_score": float(f1_score),
                    "fpr": float(fpr)
                }
            except Exception as e:
                threshold_metrics[str(threshold)] = {"error": f"Error computing metrics: {str(e)}"}
        
        # combine
        results = {
            "probability_preprocessing": transformation_message,
            "roc_curve": roc_curve_data,
            "pr_curve": pr_curve_data,
            "threshold_metrics": threshold_metrics
        }
        
        return results        
        
    def calculate_subgroup_metrics(self, csv_path: Path, 
                                categorical_features: List[str]) -> Dict[str, Any]:
        """
        Calculate metrics for subgroups based on categorical features.
        
        Parameters
        ----------
        csv_path : Path
            Path to the CSV file containing the data.
        categorical_features : List[str]
            List of categorical feature names.
            
        Returns
        -------
        dict
            Dictionary containing metrics for each subgroup.
        """
        # handle delimiter and read the CSV file
        delimiter = detect_delimiter(csv_path)
        decimal = detect_decimal(csv_path)
        df = pd.read_csv(csv_path, sep=delimiter, decimal=decimal)
        
        y_true, y_pred, valid_indices = self.prepare_data()
        
        if len(y_true) == 0:
            return {"error": "No valid numeric data pairs found for subgroup metric calculation"}
        
        model_type = self.metadata_json.get("model_type", "regression")
        metrics_module = (classification_metrics if model_type.lower() == "classification" 
                        else regression_metrics)
        
        feature_importance = self.metadata_json.get("feature_importance")
        
        valid_data = df.iloc[valid_indices].reset_index(drop=True)
        
        valid_data['y_true'] = y_true
        valid_data['y_pred'] = y_pred
        
        # if we have sensitive attribute info, add it to valid_data
        sensitive_attribute = self.metadata_json.get("sensitive_attribute")
        has_sensitive_data = False
        
        if sensitive_attribute and self.inputs:
            sensitive_values = []
            for i in valid_indices:
                if isinstance(self.inputs[i], dict) and sensitive_attribute in self.inputs[i]:
                    sensitive_values.append(self.inputs[i][sensitive_attribute])
                else:
                    sensitive_values.append(None)
            
            if any(v is not None for v in sensitive_values):
                valid_data['sensitive_attr'] = sensitive_values
                has_sensitive_data = True
        
        subgroup_metrics = {}
        
        for feature in categorical_features:
            if feature not in valid_data.columns:
                subgroup_metrics[feature] = {"error": f"Feature '{feature}' not found in data"}
                continue
                
            feature_values = valid_data[feature].unique()
            feature_metrics = {}
            
            for value in feature_values:
                # filter data for this subgroup
                subgroup_data = valid_data[valid_data[feature] == value]
                
                # skip if no data in this subgroup
                if len(subgroup_data) == 0:
                    continue
                    
                subgroup_y_true = subgroup_data['y_true'].values
                subgroup_y_pred = subgroup_data['y_pred'].values
                
                subgroup_sensitive = subgroup_data['sensitive_attr'].values if has_sensitive_data else None
                
                subgroup_all_metrics = {
                    "sample_size": len(subgroup_y_true)  # include sample size for context
                }
                
                self._compute_metrics_for_category(
                    metrics_module.performance, "performance", 
                    subgroup_y_true, subgroup_y_pred, subgroup_all_metrics
                )
                
                self._compute_metrics_for_category(
                    metrics_module.fairness, "fairness", 
                    subgroup_y_true, subgroup_y_pred, subgroup_all_metrics,
                    sensitive_values=subgroup_sensitive
                )
                
                self._compute_metrics_for_category(
                    metrics_module.explainability, "explainability", 
                    subgroup_y_true, subgroup_y_pred, subgroup_all_metrics,
                    feature_importance=feature_importance
                )
                
                feature_metrics[str(value)] = subgroup_all_metrics                
            subgroup_metrics[feature] = feature_metrics
            
        return subgroup_metrics
    
    def calculate_all_metrics_from_json(self, csv_path: Path, column_metadata_path: Path) -> Dict[str, Any]:
        """
        Calculate all metrics including overall and subgroup metrics using column metadata.
        Automatically detects if classification outputs are probabilities and calculates threshold metrics.
        
        Parameters
        ----------
        csv_path : Path
            Path to the CSV file
        column_metadata_path : Path
            Path to the column metadata JSON file
                
        Returns
        -------
        dict
            Dictionary containing all metrics.
        """
        categorical_features = self.get_categorical_features_from_json(column_metadata_path)
        
        # TODO: when the metadata is updated, we should be able to get the model type from there and remove default
        model_type = self.metadata_json.get("model_type", "classification")
        
        # calculate basic metrics first
        result = {
            "model_info": {
                "name": self.model_metadata.model_name,
                "type": model_type
            },
            "overall": self.calculate_metrics()
        }
        
        # for classification models, detect if we have probability outputs
        if model_type.lower() == "classification":
            # Get the data
            y_true, y_pred, _ = self.prepare_data()
            
            # check if predictions look like probabilities (between 0 and 1)
            is_prob_output = False
            if len(y_true) > 0:
                if np.all((y_pred >= 0) & (y_pred <= 1)):
                    # And not all values are exactly 0 or 1
                    if not np.all(np.isin(y_pred, [0, 1])):
                        is_prob_output = True
            
            # ff probability outputs, calculate threshold metrics
            if is_prob_output:
                try:
                    threshold_metrics = self.calculate_threshold_metrics()
                    if "error" not in threshold_metrics:
                        result["threshold_metrics"] = threshold_metrics
                    else:
                        result["threshold_metrics"] = {"status": "error", "message": threshold_metrics["error"]}
                except Exception as e:
                    result["threshold_metrics"] = {"status": "error", "message": f"Failed to calculate threshold metrics: {str(e)}"}
        
        # subgroup metrics if we have categorical features
        if csv_path and categorical_features:
            result["subgroups"] = self.calculate_subgroup_metrics(csv_path, categorical_features)
            
        return result

    def save_metrics_to_json_from_metadata(self, output_path: Path, csv_path: Path, 
                                        column_metadata_path: Path) -> Dict[str, Any]:
        """
        Calculate all metrics and save them to a JSON file using column metadata.
        Automatically handles probability outputs for classification models.
        
        Parameters
        ----------
        output_path : Path
            Path to save the JSON file.
        csv_path : Path
            Path to the CSV file
        column_metadata_path : Path
            Path to the column metadata JSON file
                
        Returns
        -------
        dict
            Dictionary containing all metrics.
        """
        metrics = self.calculate_all_metrics_from_json(csv_path, column_metadata_path)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics