"""
Tests for the evaluation service module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from automl_framework.services.evaluation_service import (
    ComprehensiveEvaluator, ConfusionMatrixResult, StatisticalTestResult, MetricType,
    ModelComparator, ModelComparisonResult, CrossValidationResult
)
from automl_framework.models.data_models import TaskType, PerformanceMetrics


class TestComprehensiveEvaluator:
    """Test cases for ComprehensiveEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample classification data
        self.X_clf, self.y_clf = make_classification(
            n_samples=1000, n_features=20, n_classes=3, 
            n_informative=15, random_state=42
        )
        
        # Create sample regression data
        self.X_reg, self.y_reg = make_regression(
            n_samples=1000, n_features=20, noise=0.1, random_state=42
        )
        
        # Create evaluators
        self.clf_evaluator = ComprehensiveEvaluator(
            TaskType.CLASSIFICATION, 
            class_labels=['Class_0', 'Class_1', 'Class_2']
        )
        self.reg_evaluator = ComprehensiveEvaluator(TaskType.REGRESSION)
    
    def test_init(self):
        """Test evaluator initialization."""
        evaluator = ComprehensiveEvaluator(TaskType.CLASSIFICATION)
        assert evaluator.task_type == TaskType.CLASSIFICATION
        assert evaluator.class_labels == []
        
        evaluator_with_labels = ComprehensiveEvaluator(
            TaskType.CLASSIFICATION, 
            class_labels=['A', 'B', 'C']
        )
        assert evaluator_with_labels.class_labels == ['A', 'B', 'C']
    
    def test_evaluate_classification_basic(self):
        """Test basic classification evaluation."""
        # Create simple test data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        
        metrics = self.clf_evaluator.evaluate_classification(y_true, y_pred)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.accuracy is not None
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1_score is not None
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
    
    def test_evaluate_classification_with_probabilities(self):
        """Test classification evaluation with probability scores."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
        
        metrics = self.clf_evaluator.evaluate_classification(
            y_true, y_pred, y_pred_proba
        )
        
        assert 'auc_roc' in metrics.additional_metrics
        assert 0 <= metrics.additional_metrics['auc_roc'] <= 1
    
    def test_evaluate_classification_multiclass(self):
        """Test multiclass classification evaluation."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1, 2])
        
        evaluator = ComprehensiveEvaluator(
            TaskType.CLASSIFICATION,
            class_labels=['A', 'B', 'C']
        )
        
        metrics = evaluator.evaluate_classification(y_true, y_pred)
        
        # Check per-class metrics are included
        assert 'precision_class_A' in metrics.additional_metrics
        assert 'recall_class_A' in metrics.additional_metrics
        assert 'f1_class_A' in metrics.additional_metrics
    
    def test_evaluate_classification_invalid_input(self):
        """Test classification evaluation with invalid input."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])  # Different length
        
        with pytest.raises(ValueError, match="must have the same length"):
            self.clf_evaluator.evaluate_classification(y_true, y_pred)
    
    def test_evaluate_regression_basic(self):
        """Test basic regression evaluation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
        
        metrics = self.reg_evaluator.evaluate_regression(y_true, y_pred)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.loss is not None  # MSE
        assert 'mse' in metrics.additional_metrics
        assert 'mae' in metrics.additional_metrics
        assert 'rmse' in metrics.additional_metrics
        assert 'r2_score' in metrics.additional_metrics
        assert metrics.loss >= 0
    
    def test_evaluate_regression_perfect_prediction(self):
        """Test regression evaluation with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = self.reg_evaluator.evaluate_regression(y_true, y_pred)
        
        assert metrics.loss == 0.0  # Perfect MSE
        assert metrics.additional_metrics['r2_score'] == 1.0  # Perfect R²
    
    def test_evaluate_regression_invalid_input(self):
        """Test regression evaluation with invalid input."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])  # Different length
        
        with pytest.raises(ValueError, match="must have the same length"):
            self.reg_evaluator.evaluate_regression(y_true, y_pred)
    
    def test_generate_confusion_matrix_basic(self):
        """Test basic confusion matrix generation."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        
        cm_result = self.clf_evaluator.generate_confusion_matrix(y_true, y_pred)
        
        assert isinstance(cm_result, ConfusionMatrixResult)
        assert cm_result.matrix.shape == (3, 3)
        assert len(cm_result.labels) == 3
        assert cm_result.normalized_matrix is None
        assert len(cm_result.per_class_metrics) == 3
    
    def test_generate_confusion_matrix_normalized(self):
        """Test normalized confusion matrix generation."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        
        cm_result = self.clf_evaluator.generate_confusion_matrix(
            y_true, y_pred, normalize='true'
        )
        
        assert cm_result.normalized_matrix is not None
        # Check that rows sum to 1 (normalized by true labels)
        row_sums = cm_result.normalized_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0, 1.0])
    
    def test_generate_confusion_matrix_per_class_metrics(self):
        """Test per-class metrics in confusion matrix."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])
        
        cm_result = self.clf_evaluator.generate_confusion_matrix(y_true, y_pred)
        
        # Check that per-class metrics are calculated
        for label in cm_result.labels:
            assert label in cm_result.per_class_metrics
            metrics = cm_result.per_class_metrics[label]
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'specificity' in metrics
            assert 'f1_score' in metrics
            assert 'support' in metrics
    
    def test_generate_confusion_matrix_non_classification(self):
        """Test confusion matrix with non-classification task."""
        with pytest.raises(ValueError, match="only applicable for classification"):
            self.reg_evaluator.generate_confusion_matrix(
                np.array([1, 2, 3]), np.array([1, 2, 3])
            )
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_confusion_matrix(self, mock_show, mock_savefig):
        """Test confusion matrix visualization."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        
        cm_result = self.clf_evaluator.generate_confusion_matrix(y_true, y_pred)
        
        # Test without saving
        result = self.clf_evaluator.visualize_confusion_matrix(cm_result)
        assert result is None
        mock_show.assert_called_once()
        
        # Test with saving
        mock_show.reset_mock()
        save_path = "/tmp/test_cm.png"
        result = self.clf_evaluator.visualize_confusion_matrix(cm_result, save_path)
        assert result == save_path
        mock_savefig.assert_called_once()
    
    def test_perform_statistical_test_paired_ttest(self):
        """Test paired t-test statistical testing."""
        scores1 = [0.8, 0.85, 0.82, 0.88, 0.79]
        scores2 = [0.75, 0.80, 0.78, 0.83, 0.74]
        
        result = self.clf_evaluator.perform_statistical_test(
            scores1, scores2, test_type='paired_ttest'
        )
        
        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "Paired t-test"
        assert result.p_value is not None
        assert result.statistic is not None
        assert result.effect_size is not None
        assert result.is_significant in [True, False]
    
    def test_perform_statistical_test_wilcoxon(self):
        """Test Wilcoxon signed-rank test."""
        scores1 = [0.8, 0.85, 0.82, 0.88, 0.79]
        scores2 = [0.75, 0.80, 0.78, 0.83, 0.74]
        
        result = self.clf_evaluator.perform_statistical_test(
            scores1, scores2, test_type='wilcoxon'
        )
        
        assert result.test_name == "Wilcoxon signed-rank test"
        assert result.effect_size is None  # Not calculated for Wilcoxon
    
    def test_perform_statistical_test_mannwhitney(self):
        """Test Mann-Whitney U test."""
        scores1 = [0.8, 0.85, 0.82, 0.88, 0.79]
        scores2 = [0.75, 0.80, 0.78, 0.83, 0.74, 0.77]  # Different length OK
        
        result = self.clf_evaluator.perform_statistical_test(
            scores1, scores2, test_type='mannwhitney'
        )
        
        assert result.test_name == "Mann-Whitney U test"
    
    def test_perform_statistical_test_invalid_type(self):
        """Test statistical test with invalid test type."""
        scores1 = [0.8, 0.85, 0.82]
        scores2 = [0.75, 0.80, 0.78]
        
        with pytest.raises(ValueError, match="Unsupported test type"):
            self.clf_evaluator.perform_statistical_test(
                scores1, scores2, test_type='invalid_test'
            )
    
    def test_perform_statistical_test_unequal_length_paired(self):
        """Test paired t-test with unequal length arrays."""
        scores1 = [0.8, 0.85, 0.82]
        scores2 = [0.75, 0.80]  # Different length
        
        with pytest.raises(ValueError, match="equal length score arrays"):
            self.clf_evaluator.perform_statistical_test(
                scores1, scores2, test_type='paired_ttest'
            )
    
    def test_cross_validate_model_classification(self):
        """Test cross-validation for classification model."""
        # Create and train a simple model
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_clf, self.y_clf, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        cv_results = self.clf_evaluator.cross_validate_model(
            model, X_test, y_test, cv_folds=3, scoring='accuracy'
        )
        
        assert 'cv_scores' in cv_results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert 'min_score' in cv_results
        assert 'max_score' in cv_results
        assert len(cv_results['cv_scores']) == 3
        assert 0 <= cv_results['mean_score'] <= 1
    
    def test_cross_validate_model_regression(self):
        """Test cross-validation for regression model."""
        # Create and train a simple model
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_reg, self.y_reg, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        cv_results = self.reg_evaluator.cross_validate_model(
            model, X_test, y_test, cv_folds=3, scoring='r2'
        )
        
        assert 'cv_scores' in cv_results
        assert len(cv_results['cv_scores']) == 3
        assert cv_results['scoring_metric'] == 'r2'


class TestConfusionMatrixResult:
    """Test cases for ConfusionMatrixResult class."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        matrix = np.array([[10, 2], [3, 15]])
        normalized_matrix = np.array([[0.83, 0.17], [0.17, 0.83]])
        labels = ['Class_0', 'Class_1']
        per_class_metrics = {
            'Class_0': {'precision': 0.77, 'recall': 0.83},
            'Class_1': {'precision': 0.88, 'recall': 0.83}
        }
        
        cm_result = ConfusionMatrixResult(
            matrix=matrix,
            labels=labels,
            normalized_matrix=normalized_matrix,
            per_class_metrics=per_class_metrics
        )
        
        result_dict = cm_result.to_dict()
        
        assert 'matrix' in result_dict
        assert 'labels' in result_dict
        assert 'normalized_matrix' in result_dict
        assert 'per_class_metrics' in result_dict
        assert result_dict['labels'] == labels
        assert result_dict['per_class_metrics'] == per_class_metrics


class TestStatisticalTestResult:
    """Test cases for StatisticalTestResult class."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        test_result = StatisticalTestResult(
            test_name="Paired t-test",
            statistic=2.5,
            p_value=0.03,
            is_significant=True,
            confidence_level=0.95,
            effect_size=0.8,
            interpretation="Significant difference detected"
        )
        
        result_dict = test_result.to_dict()
        
        assert result_dict['test_name'] == "Paired t-test"
        assert result_dict['statistic'] == 2.5
        assert result_dict['p_value'] == 0.03
        assert result_dict['is_significant'] is True
        assert result_dict['confidence_level'] == 0.95
        assert result_dict['effect_size'] == 0.8
        assert result_dict['interpretation'] == "Significant difference detected"


class TestModelComparator:
    """Test cases for ModelComparator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        self.X_clf, self.y_clf = make_classification(
            n_samples=200, n_features=10, n_classes=2, random_state=42
        )
        self.X_reg, self.y_reg = make_regression(
            n_samples=200, n_features=10, noise=0.1, random_state=42
        )
        
        # Create comparators
        self.clf_comparator = ModelComparator(TaskType.CLASSIFICATION)
        self.reg_comparator = ModelComparator(TaskType.REGRESSION)
        
        # Create sample model results
        self.sample_clf_results = {
            'Model_A': PerformanceMetrics(
                accuracy=0.85, precision=0.82, recall=0.88, f1_score=0.85,
                additional_metrics={'auc_roc': 0.90}
            ),
            'Model_B': PerformanceMetrics(
                accuracy=0.82, precision=0.85, recall=0.80, f1_score=0.82,
                additional_metrics={'auc_roc': 0.87}
            ),
            'Model_C': PerformanceMetrics(
                accuracy=0.88, precision=0.86, recall=0.90, f1_score=0.88,
                additional_metrics={'auc_roc': 0.92}
            )
        }
        
        self.sample_reg_results = {
            'Model_X': PerformanceMetrics(
                loss=0.15, additional_metrics={'r2_score': 0.85, 'mae': 0.12, 'rmse': 0.39}
            ),
            'Model_Y': PerformanceMetrics(
                loss=0.12, additional_metrics={'r2_score': 0.88, 'mae': 0.10, 'rmse': 0.35}
            ),
            'Model_Z': PerformanceMetrics(
                loss=0.18, additional_metrics={'r2_score': 0.82, 'mae': 0.14, 'rmse': 0.42}
            )
        }
    
    def test_compare_models_classification(self):
        """Test model comparison for classification."""
        result = self.clf_comparator.compare_models(self.sample_clf_results)
        
        assert isinstance(result, ModelComparisonResult)
        assert len(result.model_rankings) == 3
        assert result.best_model == 'Model_C'  # Should have highest composite score
        assert result.model_rankings[0]['rank'] == 1
        assert result.model_rankings[0]['model_name'] == 'Model_C'
        assert 'composite_score' in result.model_rankings[0]
    
    def test_compare_models_regression(self):
        """Test model comparison for regression."""
        result = self.reg_comparator.compare_models(self.sample_reg_results)
        
        assert isinstance(result, ModelComparisonResult)
        assert len(result.model_rankings) == 3
        assert result.best_model == 'Model_Y'  # Should have best metrics
        assert result.model_rankings[0]['rank'] == 1
        assert result.model_rankings[0]['model_name'] == 'Model_Y'
    
    def test_compare_models_custom_criteria(self):
        """Test model comparison with custom ranking criteria."""
        custom_criteria = ['accuracy', 'precision']
        custom_weights = {'accuracy': 0.7, 'precision': 0.3}
        
        result = self.clf_comparator.compare_models(
            self.sample_clf_results,
            ranking_criteria=custom_criteria,
            weights=custom_weights
        )
        
        assert result.ranking_criteria == custom_criteria
        assert len(result.model_rankings) == 3
    
    def test_compare_models_empty_input(self):
        """Test model comparison with empty input."""
        with pytest.raises(ValueError, match="At least one model result must be provided"):
            self.clf_comparator.compare_models({})
    
    def test_compare_models_with_cv(self):
        """Test model comparison with cross-validation."""
        # Create simple models
        models = {
            'RF': RandomForestClassifier(n_estimators=10, random_state=42),
            'RF2': RandomForestClassifier(n_estimators=5, random_state=42)
        }
        
        cv_results = self.clf_comparator.compare_models_with_cv(
            models, self.X_clf, self.y_clf, cv_folds=3
        )
        
        assert len(cv_results) == 2
        assert 'RF' in cv_results
        assert 'RF2' in cv_results
        
        for model_name, cv_result in cv_results.items():
            assert isinstance(cv_result, CrossValidationResult)
            assert cv_result.model_name == model_name
            assert len(cv_result.cv_scores) == 3
            assert cv_result.cv_folds == 3
            assert len(cv_result.confidence_interval) == 2
    
    def test_perform_model_significance_tests(self):
        """Test statistical significance testing between models."""
        # Create mock CV results
        cv_results = {
            'Model_A': CrossValidationResult(
                model_name='Model_A',
                cv_scores=[0.8, 0.85, 0.82, 0.88, 0.79],
                mean_score=0.828,
                std_score=0.035,
                confidence_interval=(0.79, 0.87),
                scoring_metric='accuracy',
                cv_folds=5
            ),
            'Model_B': CrossValidationResult(
                model_name='Model_B',
                cv_scores=[0.75, 0.80, 0.78, 0.83, 0.74],
                mean_score=0.78,
                std_score=0.037,
                confidence_interval=(0.74, 0.82),
                scoring_metric='accuracy',
                cv_folds=5
            )
        }
        
        test_results = self.clf_comparator.perform_model_significance_tests(cv_results)
        
        assert len(test_results) == 1  # Only one comparison for 2 models
        assert 'Model_A_vs_Model_B' in test_results
        
        test_result = test_results['Model_A_vs_Model_B']
        assert isinstance(test_result, StatisticalTestResult)
        assert test_result.p_value is not None
        assert test_result.statistic is not None
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        comparison_result = self.clf_comparator.compare_models(self.sample_clf_results)
        
        report = self.clf_comparator.generate_performance_report(comparison_result)
        
        assert isinstance(report, str)
        assert "MODEL PERFORMANCE COMPARISON REPORT" in report
        assert "BEST MODEL: Model_C" in report
        assert "MODEL RANKINGS:" in report
        assert "DETAILED METRICS COMPARISON:" in report
    
    def test_generate_performance_report_with_cv(self):
        """Test performance report generation with CV results."""
        comparison_result = self.clf_comparator.compare_models(self.sample_clf_results)
        
        # Create mock CV results
        cv_results = {
            'Model_A': CrossValidationResult(
                model_name='Model_A',
                cv_scores=[0.8, 0.85, 0.82],
                mean_score=0.823,
                std_score=0.025,
                confidence_interval=(0.79, 0.86),
                scoring_metric='accuracy',
                cv_folds=3
            )
        }
        
        report = self.clf_comparator.generate_performance_report(
            comparison_result, cv_results
        )
        
        assert "CROSS-VALIDATION RESULTS:" in report
        assert "Mean Score:" in report
        assert "95% CI:" in report
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_model_comparison(self, mock_show, mock_savefig):
        """Test model comparison visualization."""
        comparison_result = self.clf_comparator.compare_models(self.sample_clf_results)
        
        # Test without saving
        result = self.clf_comparator.visualize_model_comparison(comparison_result)
        assert result is None
        mock_show.assert_called_once()
        
        # Test with saving
        mock_show.reset_mock()
        save_path = "/tmp/test_comparison.png"
        result = self.clf_comparator.visualize_model_comparison(
            comparison_result, save_path=save_path
        )
        assert result == save_path
        mock_savefig.assert_called_once()
    
    def test_bootstrap_model_evaluation(self):
        """Test bootstrap evaluation of a model."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        bootstrap_result = self.clf_comparator.bootstrap_model_evaluation(
            model, self.X_clf, self.y_clf, n_bootstrap=50, random_state=42
        )
        
        assert 'bootstrap_scores' in bootstrap_result
        assert 'mean_score' in bootstrap_result
        assert 'std_score' in bootstrap_result
        assert 'confidence_intervals' in bootstrap_result
        assert len(bootstrap_result['bootstrap_scores']) == 50
        
        # Check confidence intervals
        ci = bootstrap_result['confidence_intervals']
        assert '90%' in ci
        assert '95%' in ci
        assert '99%' in ci
        
        for level, (lower, upper) in ci.items():
            assert lower <= upper
            assert 0 <= lower <= 1
            assert 0 <= upper <= 1


class TestModelComparisonResult:
    """Test cases for ModelComparisonResult class."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Create sample data
        rankings = [
            {'rank': 1, 'model_name': 'Model_A', 'composite_score': 0.85},
            {'rank': 2, 'model_name': 'Model_B', 'composite_score': 0.80}
        ]
        
        comparison_matrix = pd.DataFrame({
            'accuracy': [0.85, 0.80],
            'precision': [0.82, 0.85]
        }, index=['Model_A', 'Model_B'])
        
        statistical_tests = {
            'Model_A_vs_Model_B': StatisticalTestResult(
                test_name="Paired t-test",
                statistic=2.5,
                p_value=0.03,
                is_significant=True
            )
        }
        
        result = ModelComparisonResult(
            model_rankings=rankings,
            comparison_matrix=comparison_matrix,
            statistical_tests=statistical_tests,
            best_model='Model_A',
            ranking_criteria=['accuracy', 'precision']
        )
        
        result_dict = result.to_dict()
        
        assert 'model_rankings' in result_dict
        assert 'comparison_matrix' in result_dict
        assert 'statistical_tests' in result_dict
        assert 'best_model' in result_dict
        assert 'ranking_criteria' in result_dict
        assert result_dict['best_model'] == 'Model_A'


class TestCrossValidationResult:
    """Test cases for CrossValidationResult class."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        cv_result = CrossValidationResult(
            model_name='Test_Model',
            cv_scores=[0.8, 0.85, 0.82],
            mean_score=0.823,
            std_score=0.025,
            confidence_interval=(0.79, 0.86),
            scoring_metric='accuracy',
            cv_folds=3
        )
        
        result_dict = cv_result.to_dict()
        
        assert result_dict['model_name'] == 'Test_Model'
        assert result_dict['cv_scores'] == [0.8, 0.85, 0.82]
        assert result_dict['mean_score'] == 0.823
        assert result_dict['std_score'] == 0.025
        assert result_dict['confidence_interval'] == (0.79, 0.86)
        assert result_dict['scoring_metric'] == 'accuracy'
        assert result_dict['cv_folds'] == 3