"""
Model evaluation service for the AutoML framework.

This module provides comprehensive evaluation metrics, model comparison,
and statistical analysis capabilities for trained models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from scipy import stats
import warnings
from pathlib import Path

from ..models.data_models import PerformanceMetrics, TaskType


class MetricType(Enum):
    """Types of evaluation metrics."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    CUSTOM = "custom"


@dataclass
class ConfusionMatrixResult:
    """Results from confusion matrix analysis."""
    matrix: np.ndarray
    labels: List[str]
    normalized_matrix: Optional[np.ndarray] = None
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'matrix': self.matrix.tolist(),
            'labels': self.labels,
            'normalized_matrix': self.normalized_matrix.tolist() if self.normalized_matrix is not None else None,
            'per_class_metrics': self.per_class_metrics
        }


@dataclass
class StatisticalTestResult:
    """Results from statistical significance testing."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float = 0.95
    effect_size: Optional[float] = None
    interpretation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'confidence_level': self.confidence_level,
            'effect_size': self.effect_size,
            'interpretation': self.interpretation
        }


class ComprehensiveEvaluator:
    """
    Comprehensive model evaluation with standard ML metrics.
    
    Provides evaluation methods for classification, regression, and other tasks
    with support for confusion matrices and statistical significance testing.
    """
    
    def __init__(self, task_type: TaskType, class_labels: Optional[List[str]] = None):
        """
        Initialize the evaluator.
        
        Args:
            task_type: Type of ML task (classification, regression, etc.)
            class_labels: Labels for classification tasks
        """
        self.task_type = task_type
        self.class_labels = class_labels or []
        self._supported_tasks = {
            TaskType.CLASSIFICATION,
            TaskType.REGRESSION,
            TaskType.OBJECT_DETECTION,
            TaskType.TIME_SERIES_FORECASTING
        }
        
        if task_type not in self._supported_tasks:
            warnings.warn(f"Task type {task_type} may not be fully supported")
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> PerformanceMetrics:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            average: Averaging strategy for multi-class metrics
            
        Returns:
            PerformanceMetrics with classification metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Additional metrics
        additional_metrics = {}
        
        # AUC-ROC for binary/multi-class
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
                else:  # Multi-class
                    auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
                additional_metrics['auc_roc'] = auc_roc
            except (ValueError, IndexError):
                warnings.warn("Could not compute AUC-ROC score")
        
        # Per-class metrics
        if len(np.unique(y_true)) > 2:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i, label in enumerate(self.class_labels or range(len(precision_per_class))):
                additional_metrics[f'precision_class_{label}'] = precision_per_class[i]
                additional_metrics[f'recall_class_{label}'] = recall_per_class[i]
                additional_metrics[f'f1_class_{label}'] = f1_per_class[i]
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            additional_metrics=additional_metrics
        )
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> PerformanceMetrics:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            PerformanceMetrics with regression metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Additional metrics
        additional_metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2
        }
        
        # Mean Absolute Percentage Error (MAPE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            if not np.isnan(mape) and not np.isinf(mape):
                additional_metrics['mape'] = mape
        
        # Explained variance score
        from sklearn.metrics import explained_variance_score
        explained_var = explained_variance_score(y_true, y_pred)
        additional_metrics['explained_variance'] = explained_var
        
        return PerformanceMetrics(
            loss=mse,  # Use MSE as primary loss metric
            additional_metrics=additional_metrics
        )
    
    def generate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> ConfusionMatrixResult:
        """
        Generate confusion matrix with detailed analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization method ('true', 'pred', 'all', or None)
            
        Returns:
            ConfusionMatrixResult with matrix and per-class metrics
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("Confusion matrix is only applicable for classification tasks")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get unique labels
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        labels = [str(label) for label in unique_labels]
        
        # Normalize if requested
        normalized_cm = None
        if normalize:
            normalized_cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(labels):
            if i < len(cm):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                per_class_metrics[label] = {
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'f1_score': f1,
                    'support': int(cm[i, :].sum())
                }
        
        return ConfusionMatrixResult(
            matrix=cm,
            labels=labels,
            normalized_matrix=normalized_cm,
            per_class_metrics=per_class_metrics
        )
    
    def visualize_confusion_matrix(
        self,
        confusion_result: ConfusionMatrixResult,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Optional[str]:
        """
        Create visualization of confusion matrix.
        
        Args:
            confusion_result: Result from generate_confusion_matrix
            save_path: Path to save the plot (optional)
            figsize: Figure size for the plot
            
        Returns:
            Path to saved plot if save_path provided, None otherwise
        """
        plt.figure(figsize=figsize)
        
        # Use normalized matrix if available, otherwise use raw counts
        matrix_to_plot = (confusion_result.normalized_matrix 
                         if confusion_result.normalized_matrix is not None 
                         else confusion_result.matrix)
        
        # Create heatmap
        sns.heatmap(
            matrix_to_plot,
            annot=True,
            fmt='.2f' if confusion_result.normalized_matrix is not None else 'd',
            cmap='Blues',
            xticklabels=confusion_result.labels,
            yticklabels=confusion_result.labels,
            cbar_kws={'label': 'Normalized Count' if confusion_result.normalized_matrix is not None else 'Count'}
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def perform_statistical_test(
        self,
        scores1: List[float],
        scores2: List[float],
        test_type: str = 'paired_ttest',
        confidence_level: float = 0.95
    ) -> StatisticalTestResult:
        """
        Perform statistical significance testing between two sets of scores.
        
        Args:
            scores1: First set of performance scores
            scores2: Second set of performance scores
            test_type: Type of statistical test ('paired_ttest', 'wilcoxon', 'mannwhitney')
            confidence_level: Confidence level for the test
            
        Returns:
            StatisticalTestResult with test results
        """
        if len(scores1) != len(scores2) and test_type == 'paired_ttest':
            raise ValueError("Paired t-test requires equal length score arrays")
        
        alpha = 1 - confidence_level
        
        if test_type == 'paired_ttest':
            statistic, p_value = stats.ttest_rel(scores1, scores2)
            effect_size = (np.mean(scores1) - np.mean(scores2)) / np.std(np.array(scores1) - np.array(scores2))
            test_name = "Paired t-test"
            
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(scores1, scores2)
            effect_size = None  # Effect size not standard for Wilcoxon
            test_name = "Wilcoxon signed-rank test"
            
        elif test_type == 'mannwhitney':
            statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
            effect_size = None  # Effect size calculation would require additional computation
            test_name = "Mann-Whitney U test"
            
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        is_significant = p_value < alpha
        
        # Generate interpretation
        if is_significant:
            interpretation = f"Significant difference detected (p={p_value:.4f} < α={alpha:.3f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f} ≥ α={alpha:.3f})"
        
        return StatisticalTestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=confidence_level,
            effect_size=effect_size,
            interpretation=interpretation
        )
    
    def cross_validate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: str = 'accuracy',
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Trained model with fit/predict methods
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for evaluation
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with cross-validation results
        """
        # Choose appropriate cross-validation strategy
        if self.task_type == TaskType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(np.mean(cv_scores)),
            'std_score': float(np.std(cv_scores)),
            'min_score': float(np.min(cv_scores)),
            'max_score': float(np.max(cv_scores)),
            'scoring_metric': scoring,
            'cv_folds': cv_folds
        }


@dataclass
class ModelComparisonResult:
    """Results from model comparison analysis."""
    model_rankings: List[Dict[str, Any]]
    comparison_matrix: pd.DataFrame
    statistical_tests: Dict[str, StatisticalTestResult]
    best_model: str
    ranking_criteria: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_rankings': self.model_rankings,
            'comparison_matrix': self.comparison_matrix.to_dict(),
            'statistical_tests': {k: v.to_dict() for k, v in self.statistical_tests.items()},
            'best_model': self.best_model,
            'ranking_criteria': self.ranking_criteria
        }


@dataclass
class CrossValidationResult:
    """Results from cross-validation evaluation."""
    model_name: str
    cv_scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    scoring_metric: str
    cv_folds: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'cv_scores': self.cv_scores,
            'mean_score': self.mean_score,
            'std_score': self.std_score,
            'confidence_interval': self.confidence_interval,
            'scoring_metric': self.scoring_metric,
            'cv_folds': self.cv_folds
        }


class ModelComparator:
    """
    Model comparison and ranking service.
    
    Provides functionality to compare multiple models, rank them based on
    various criteria, and perform statistical significance testing.
    """
    
    def __init__(self, task_type: TaskType):
        """
        Initialize the model comparator.
        
        Args:
            task_type: Type of ML task (classification, regression, etc.)
        """
        self.task_type = task_type
        self.evaluator = ComprehensiveEvaluator(task_type)
    
    def compare_models(
        self,
        model_results: Dict[str, PerformanceMetrics],
        ranking_criteria: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> ModelComparisonResult:
        """
        Compare multiple models and rank them.
        
        Args:
            model_results: Dictionary mapping model names to their performance metrics
            ranking_criteria: List of metrics to use for ranking
            weights: Weights for each ranking criterion
            
        Returns:
            ModelComparisonResult with rankings and comparisons
        """
        if not model_results:
            raise ValueError("At least one model result must be provided")
        
        # Set default ranking criteria based on task type
        if ranking_criteria is None:
            if self.task_type == TaskType.CLASSIFICATION:
                ranking_criteria = ['accuracy', 'f1_score', 'precision', 'recall']
            elif self.task_type == TaskType.REGRESSION:
                ranking_criteria = ['r2_score', 'mae', 'rmse']  # Note: lower is better for mae, rmse
            else:
                ranking_criteria = ['accuracy']  # Default fallback
        
        # Set default weights (equal weighting)
        if weights is None:
            weights = {criterion: 1.0 for criterion in ranking_criteria}
        
        # Create comparison matrix
        comparison_data = []
        for model_name, metrics in model_results.items():
            row = {'model_name': model_name}
            
            # Add primary metrics
            if metrics.accuracy is not None:
                row['accuracy'] = metrics.accuracy
            if metrics.precision is not None:
                row['precision'] = metrics.precision
            if metrics.recall is not None:
                row['recall'] = metrics.recall
            if metrics.f1_score is not None:
                row['f1_score'] = metrics.f1_score
            if metrics.loss is not None:
                row['loss'] = metrics.loss
            
            # Add additional metrics
            for metric_name, value in metrics.additional_metrics.items():
                row[metric_name] = value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('model_name', inplace=True)
        
        # Calculate composite scores for ranking
        model_scores = self._calculate_composite_scores(
            comparison_df, ranking_criteria, weights
        )
        
        # Create rankings
        rankings = []
        for rank, (model_name, score) in enumerate(model_scores, 1):
            model_info = {
                'rank': rank,
                'model_name': model_name,
                'composite_score': score,
                'metrics': comparison_df.loc[model_name].to_dict()
            }
            rankings.append(model_info)
        
        # Perform statistical tests (if applicable)
        statistical_tests = {}
        if len(model_results) >= 2:
            # This would require cross-validation scores for proper statistical testing
            # For now, we'll create placeholder tests
            model_names = list(model_results.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    test_key = f"{model_names[i]}_vs_{model_names[j]}"
                    # Placeholder - in practice, this would use CV scores
                    statistical_tests[test_key] = StatisticalTestResult(
                        test_name="Model Comparison (Placeholder)",
                        statistic=0.0,
                        p_value=1.0,
                        is_significant=False,
                        interpretation="Statistical test requires cross-validation scores"
                    )
        
        best_model = rankings[0]['model_name'] if rankings else ""
        
        return ModelComparisonResult(
            model_rankings=rankings,
            comparison_matrix=comparison_df,
            statistical_tests=statistical_tests,
            best_model=best_model,
            ranking_criteria=ranking_criteria
        )
    
    def _calculate_composite_scores(
        self,
        comparison_df: pd.DataFrame,
        criteria: List[str],
        weights: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Calculate composite scores for model ranking.
        
        Args:
            comparison_df: DataFrame with model metrics
            criteria: List of criteria to use for scoring
            weights: Weights for each criterion
            
        Returns:
            List of (model_name, composite_score) tuples, sorted by score (descending)
        """
        # Metrics where lower values are better
        lower_is_better = {'loss', 'mse', 'mae', 'rmse', 'mape'}
        
        model_scores = {}
        
        for model_name in comparison_df.index:
            composite_score = 0.0
            total_weight = 0.0
            
            for criterion in criteria:
                if criterion in comparison_df.columns and not pd.isna(comparison_df.loc[model_name, criterion]):
                    value = comparison_df.loc[model_name, criterion]
                    weight = weights.get(criterion, 1.0)
                    
                    # Normalize the score (higher is better)
                    if criterion in lower_is_better:
                        # For metrics where lower is better, invert the score
                        # Use 1 / (1 + value) to ensure positive scores
                        normalized_score = 1.0 / (1.0 + value)
                    else:
                        # For metrics where higher is better, use as-is
                        normalized_score = value
                    
                    composite_score += normalized_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                model_scores[model_name] = composite_score / total_weight
            else:
                model_scores[model_name] = 0.0
        
        # Sort by composite score (descending)
        return sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    def compare_models_with_cv(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: str = 'accuracy',
        random_state: Optional[int] = None
    ) -> Dict[str, CrossValidationResult]:
        """
        Compare models using cross-validation.
        
        Args:
            models: Dictionary mapping model names to model instances
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for evaluation
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary mapping model names to CrossValidationResult
        """
        cv_results = {}
        
        for model_name, model in models.items():
            cv_result = self.evaluator.cross_validate_model(
                model, X, y, cv_folds, scoring, random_state
            )
            
            # Calculate confidence interval
            scores = cv_result['cv_scores']
            mean_score = cv_result['mean_score']
            std_score = cv_result['std_score']
            
            # 95% confidence interval using t-distribution
            from scipy.stats import t
            confidence_level = 0.95
            alpha = 1 - confidence_level
            dof = len(scores) - 1
            t_value = t.ppf(1 - alpha/2, dof)
            margin_error = t_value * (std_score / np.sqrt(len(scores)))
            
            confidence_interval = (
                mean_score - margin_error,
                mean_score + margin_error
            )
            
            cv_results[model_name] = CrossValidationResult(
                model_name=model_name,
                cv_scores=scores,
                mean_score=mean_score,
                std_score=std_score,
                confidence_interval=confidence_interval,
                scoring_metric=scoring,
                cv_folds=cv_folds
            )
        
        return cv_results
    
    def perform_model_significance_tests(
        self,
        cv_results: Dict[str, CrossValidationResult],
        test_type: str = 'paired_ttest',
        confidence_level: float = 0.95
    ) -> Dict[str, StatisticalTestResult]:
        """
        Perform statistical significance tests between models.
        
        Args:
            cv_results: Cross-validation results for each model
            test_type: Type of statistical test to perform
            confidence_level: Confidence level for the tests
            
        Returns:
            Dictionary mapping comparison pairs to test results
        """
        statistical_tests = {}
        model_names = list(cv_results.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                scores1 = cv_results[model1].cv_scores
                scores2 = cv_results[model2].cv_scores
                
                test_result = self.evaluator.perform_statistical_test(
                    scores1, scores2, test_type, confidence_level
                )
                
                test_key = f"{model1}_vs_{model2}"
                statistical_tests[test_key] = test_result
        
        return statistical_tests
    
    def generate_performance_report(
        self,
        comparison_result: ModelComparisonResult,
        cv_results: Optional[Dict[str, CrossValidationResult]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            comparison_result: Results from model comparison
            cv_results: Cross-validation results (optional)
            save_path: Path to save the report (optional)
            
        Returns:
            Report as a string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("MODEL PERFORMANCE COMPARISON REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Best model summary
        report_lines.append(f"BEST MODEL: {comparison_result.best_model}")
        report_lines.append("")
        
        # Model rankings
        report_lines.append("MODEL RANKINGS:")
        report_lines.append("-" * 40)
        for ranking in comparison_result.model_rankings:
            report_lines.append(
                f"{ranking['rank']}. {ranking['model_name']} "
                f"(Score: {ranking['composite_score']:.4f})"
            )
        report_lines.append("")
        
        # Detailed metrics comparison
        report_lines.append("DETAILED METRICS COMPARISON:")
        report_lines.append("-" * 40)
        report_lines.append(comparison_result.comparison_matrix.to_string())
        report_lines.append("")
        
        # Cross-validation results (if available)
        if cv_results:
            report_lines.append("CROSS-VALIDATION RESULTS:")
            report_lines.append("-" * 40)
            for model_name, cv_result in cv_results.items():
                report_lines.append(f"{model_name}:")
                report_lines.append(f"  Mean Score: {cv_result.mean_score:.4f} ± {cv_result.std_score:.4f}")
                report_lines.append(f"  95% CI: [{cv_result.confidence_interval[0]:.4f}, {cv_result.confidence_interval[1]:.4f}]")
                report_lines.append("")
        
        # Statistical significance tests
        if comparison_result.statistical_tests:
            report_lines.append("STATISTICAL SIGNIFICANCE TESTS:")
            report_lines.append("-" * 40)
            for test_name, test_result in comparison_result.statistical_tests.items():
                report_lines.append(f"{test_name}:")
                report_lines.append(f"  {test_result.interpretation}")
                report_lines.append(f"  p-value: {test_result.p_value:.4f}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def visualize_model_comparison(
        self,
        comparison_result: ModelComparisonResult,
        cv_results: Optional[Dict[str, CrossValidationResult]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> Optional[str]:
        """
        Create visualizations for model comparison.
        
        Args:
            comparison_result: Results from model comparison
            cv_results: Cross-validation results (optional)
            save_path: Path to save the plot (optional)
            figsize: Figure size for the plot
            
        Returns:
            Path to saved plot if save_path provided, None otherwise
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # 1. Model rankings bar chart
        rankings = comparison_result.model_rankings
        model_names = [r['model_name'] for r in rankings]
        scores = [r['composite_score'] for r in rankings]
        
        axes[0, 0].bar(model_names, scores, color='skyblue')
        axes[0, 0].set_title('Model Rankings (Composite Score)')
        axes[0, 0].set_ylabel('Composite Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Metrics heatmap
        metrics_df = comparison_result.comparison_matrix
        if not metrics_df.empty:
            sns.heatmap(
                metrics_df.select_dtypes(include=[np.number]),
                annot=True, fmt='.3f', cmap='RdYlBu_r',
                ax=axes[0, 1], cbar_kws={'label': 'Metric Value'}
            )
            axes[0, 1].set_title('Metrics Heatmap')
        
        # 3. Cross-validation box plot (if available)
        if cv_results:
            cv_data = []
            cv_labels = []
            for model_name, cv_result in cv_results.items():
                cv_data.append(cv_result.cv_scores)
                cv_labels.append(model_name)
            
            axes[1, 0].boxplot(cv_data, labels=cv_labels)
            axes[1, 0].set_title('Cross-Validation Score Distribution')
            axes[1, 0].set_ylabel('CV Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No CV Results Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Cross-Validation Results')
        
        # 4. Performance radar chart (for top 3 models)
        top_models = rankings[:3]
        if len(top_models) > 0 and not metrics_df.empty:
            # Remove the current axes and create a polar subplot
            fig.delaxes(axes[1, 1])
            ax_polar = fig.add_subplot(2, 2, 4, projection='polar')
            self._create_radar_chart(ax_polar, top_models, metrics_df)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient Data for Radar Chart', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Performance Radar Chart')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def _create_radar_chart(self, ax, top_models: List[Dict], metrics_df: pd.DataFrame):
        """Create a radar chart for top models."""
        # Select numeric metrics for radar chart
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            ax.text(0.5, 0.5, 'No Numeric Metrics Available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Limit to first 6 metrics for readability
        metrics_to_plot = numeric_cols[:6]
        
        # Normalize metrics to 0-1 scale for radar chart
        normalized_df = metrics_df[metrics_to_plot].copy()
        for col in metrics_to_plot:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max > col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0.5  # If all values are the same
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        colors = ['blue', 'red', 'green']
        for i, model_info in enumerate(top_models):
            model_name = model_info['model_name']
            if model_name in normalized_df.index:
                values = normalized_df.loc[model_name, metrics_to_plot].values
                values = np.concatenate((values, [values[0]]))  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=model_name, color=colors[i % len(colors)])
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_plot)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart (Top 3 Models)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    def bootstrap_model_evaluation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 1000,
        test_size: float = 0.3,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform bootstrap evaluation of a model.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            n_bootstrap: Number of bootstrap samples
            test_size: Proportion of data to use for testing in each bootstrap
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with bootstrap evaluation results
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        bootstrap_scores = []
        n_samples = len(X)
        
        for _ in range(n_bootstrap):
            # Create bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Split bootstrap sample
            X_train, X_test, y_train, y_test = train_test_split(
                X_bootstrap, y_bootstrap, test_size=test_size, random_state=None
            )
            
            # Train and evaluate model
            model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            model_copy.fit(X_train, y_train)
            
            if self.task_type == TaskType.CLASSIFICATION:
                from sklearn.metrics import accuracy_score
                y_pred = model_copy.predict(X_test)
                score = accuracy_score(y_test, y_pred)
            else:  # Regression
                from sklearn.metrics import r2_score
                y_pred = model_copy.predict(X_test)
                score = r2_score(y_test, y_pred)
            
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate statistics
        mean_score = np.mean(bootstrap_scores)
        std_score = np.std(bootstrap_scores)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for confidence_level in [0.90, 0.95, 0.99]:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_scores, lower_percentile)
            ci_upper = np.percentile(bootstrap_scores, upper_percentile)
            
            confidence_intervals[f'{confidence_level:.0%}'] = (ci_lower, ci_upper)
        
        return {
            'bootstrap_scores': bootstrap_scores.tolist(),
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'confidence_intervals': confidence_intervals,
            'n_bootstrap': n_bootstrap,
            'test_size': test_size
        }