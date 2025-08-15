"""
Performance benchmarks and regression tests for AutoML framework.

Tests performance characteristics of key components and ensures
no significant performance regressions occur.
"""

import pytest
import time
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from automl_framework.services.data_processing import DataProcessingService
from automl_framework.services.nas_service import NASService
from automl_framework.services.hyperparameter_optimization import HyperparameterOptimizer
from automl_framework.services.training_service import TrainingService
from automl_framework.services.evaluation_service import EvaluationService
from automl_framework.models.data_models import TaskType

from tests.test_utils import (
    MockDatasetGenerator, MockArchitectureGenerator, PerformanceBenchmark,
    TestDataManager, test_data_manager
)


class TestDataProcessingPerformance:
    """Performance tests for data processing service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataProcessingService()
        self.benchmark = PerformanceBenchmark()
    
    def test_dataset_analysis_performance(self, test_data_manager):
        """Test dataset analysis performance with various dataset sizes."""
        dataset_sizes = [1000, 5000, 10000, 50000]
        performance_results = {}
        
        for size in dataset_sizes:
            # Create dataset
            df, temp_file = test_data_manager.create_tabular_dataset(
                n_samples=size,
                n_features=20,
                n_classes=5
            )
            
            # Benchmark analysis
            result = self.benchmark.benchmark_function(
                self.service._analyze_dataset_internal,
                temp_file
            )
            
            performance_results[size] = {
                'execution_time': result['execution_time'],
                'peak_memory_mb': result['peak_memory_mb']
            }
            
            # Performance assertions
            assert result['execution_time'] < size * 0.001  # Should be roughly linear
            assert result['peak_memory_mb'] < size * 0.01  # Memory should scale reasonably
        
        # Check that performance scales reasonably
        times = [performance_results[size]['execution_time'] for size in dataset_sizes]
        
        # Execution time should not grow exponentially
        for i in range(1, len(times)):
            growth_factor = times[i] / times[i-1]
            size_factor = dataset_sizes[i] / dataset_sizes[i-1]
            assert growth_factor <= size_factor * 2  # Allow some overhead
    
    def test_preprocessing_pipeline_performance(self, test_data_manager):
        """Test preprocessing pipeline performance."""
        # Create large dataset
        df, temp_file = test_data_manager.create_tabular_dataset(
            n_samples=10000,
            n_features=50,
            n_classes=10,
            missing_rate=0.2,
            categorical_features=10
        )
        
        # Benchmark preprocessing
        result = self.benchmark.benchmark_function(
            self.service.process_dataset,
            temp_file,
            task_type=TaskType.CLASSIFICATION,
            enable_feature_engineering=True
        )
        
        # Performance assertions
        assert result['execution_time'] < 60.0  # Should complete within 1 minute
        assert result['peak_memory_mb'] < 1000  # Should use less than 1GB memory
        
        # Verify output quality
        processed_data = result['result']
        assert 'X_train' in processed_data
        assert 'y_train' in processed_data
        assert len(processed_data['X_train']) > 0
    
    def test_feature_engineering_performance(self, test_data_manager):
        """Test feature engineering performance."""
        # Create dataset with various feature types
        df, temp_file = test_data_manager.create_tabular_dataset(
            n_samples=5000,
            n_features=30,
            categorical_features=10
        )
        
        # Add time-based features
        df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='H')
        df['text'] = ['sample text ' + str(i) for i in range(len(df))]
        
        # Save updated dataset
        df.to_csv(temp_file, index=False)
        
        # Benchmark feature engineering
        result = self.benchmark.benchmark_function(
            self.service.process_dataset,
            temp_file,
            enable_feature_engineering=True,
            max_features=100
        )
        
        # Performance assertions
        assert result['execution_time'] < 120.0  # Should complete within 2 minutes
        assert result['peak_memory_mb'] < 2000  # Should use less than 2GB memory
        
        # Verify feature engineering worked
        processed_data = result['result']
        assert len(processed_data['feature_names']) <= 100


class TestNASPerformance:
    """Performance tests for Neural Architecture Search."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = NASService()
        self.benchmark = PerformanceBenchmark()
    
    @patch('automl_framework.services.nas_service.torch')
    def test_architecture_encoding_performance(self, mock_torch):
        """Test architecture encoding/decoding performance."""
        # Create multiple architectures
        architectures = []
        for i in range(100):
            if i % 2 == 0:
                arch = MockArchitectureGenerator.create_simple_mlp(
                    hidden_sizes=[128, 64, 32]
                )
            else:
                arch = MockArchitectureGenerator.create_simple_cnn(
                    conv_filters=[32, 64, 128],
                    dense_units=[256, 128]
                )
            architectures.append(arch)
        
        # Benchmark encoding
        encoding_result = self.benchmark.benchmark_function(
            self._encode_architectures,
            architectures
        )
        
        # Performance assertions
        assert encoding_result['execution_time'] < 5.0  # Should be fast
        assert encoding_result['peak_memory_mb'] < 100  # Should be memory efficient
        
        # Verify encoding worked
        encoded_archs = encoding_result['result']
        assert len(encoded_archs) == len(architectures)
    
    def _encode_architectures(self, architectures):
        """Helper method to encode multiple architectures."""
        encoded = []
        for arch in architectures:
            # Mock encoding process
            encoded_arch = {
                'id': arch.id,
                'layers': len(arch.layers),
                'parameters': arch.parameter_count,
                'flops': arch.flops
            }
            encoded.append(encoded_arch)
        return encoded
    
    def test_architecture_validation_performance(self):
        """Test architecture validation performance."""
        # Create many architectures to validate
        architectures = []
        for i in range(500):
            arch = MockArchitectureGenerator.create_simple_mlp(
                hidden_sizes=[64, 32] if i % 2 == 0 else [128, 64, 32]
            )
            architectures.append(arch)
        
        # Benchmark validation
        validation_result = self.benchmark.benchmark_function(
            self._validate_architectures,
            architectures
        )
        
        # Performance assertions
        assert validation_result['execution_time'] < 10.0  # Should validate quickly
        assert validation_result['peak_memory_mb'] < 200
        
        # Verify validation worked
        validation_results = validation_result['result']
        assert len(validation_results) == len(architectures)
        assert all(result['is_valid'] for result in validation_results)
    
    def _validate_architectures(self, architectures):
        """Helper method to validate multiple architectures."""
        results = []
        for arch in architectures:
            # Mock validation
            is_valid = (
                len(arch.layers) > 0 and
                arch.parameter_count > 0 and
                arch.input_shape is not None and
                arch.output_shape is not None
            )
            results.append({
                'architecture_id': arch.id,
                'is_valid': is_valid,
                'validation_time': 0.001  # Mock validation time
            })
        return results


class TestHyperparameterOptimizationPerformance:
    """Performance tests for hyperparameter optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = HyperparameterOptimizer()
        self.benchmark = PerformanceBenchmark()
    
    def test_bayesian_optimization_performance(self):
        """Test Bayesian optimization performance."""
        # Mock objective function
        def mock_objective(params):
            # Simulate some computation time
            time.sleep(0.01)
            # Return mock performance based on learning rate
            lr = params.get('learning_rate', 0.001)
            return 0.9 - abs(lr - 0.001) * 100  # Best at lr=0.001
        
        # Define search space
        search_space = {
            'learning_rate': (1e-5, 1e-1),
            'batch_size': [16, 32, 64, 128],
            'optimizer': ['adam', 'sgd', 'rmsprop']
        }
        
        # Benchmark optimization
        optimization_result = self.benchmark.benchmark_function(
            self._run_bayesian_optimization,
            mock_objective,
            search_space,
            n_trials=20
        )
        
        # Performance assertions
        assert optimization_result['execution_time'] < 30.0  # Should complete reasonably fast
        assert optimization_result['peak_memory_mb'] < 500
        
        # Verify optimization worked
        best_params = optimization_result['result']
        assert 'learning_rate' in best_params
        assert 'batch_size' in best_params
        assert 'optimizer' in best_params
    
    def _run_bayesian_optimization(self, objective_func, search_space, n_trials):
        """Helper method to run Bayesian optimization."""
        # Mock Bayesian optimization
        best_score = float('-inf')
        best_params = None
        
        for trial in range(n_trials):
            # Sample parameters
            params = {}
            params['learning_rate'] = np.random.uniform(1e-5, 1e-1)
            params['batch_size'] = np.random.choice([16, 32, 64, 128])
            params['optimizer'] = np.random.choice(['adam', 'sgd', 'rmsprop'])
            
            # Evaluate
            score = objective_func(params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params
    
    def test_hyperparameter_space_sampling_performance(self):
        """Test hyperparameter space sampling performance."""
        # Create large search space
        search_space = {}
        for i in range(50):  # Many parameters
            search_space[f'param_{i}'] = (0.0, 1.0)
        
        # Benchmark sampling
        sampling_result = self.benchmark.benchmark_function(
            self._sample_hyperparameter_space,
            search_space,
            n_samples=1000
        )
        
        # Performance assertions
        assert sampling_result['execution_time'] < 5.0  # Should sample quickly
        assert sampling_result['peak_memory_mb'] < 100
        
        # Verify sampling worked
        samples = sampling_result['result']
        assert len(samples) == 1000
        assert all(len(sample) == 50 for sample in samples)
    
    def _sample_hyperparameter_space(self, search_space, n_samples):
        """Helper method to sample hyperparameter space."""
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param_name, param_range in search_space.items():
                sample[param_name] = np.random.uniform(param_range[0], param_range[1])
            samples.append(sample)
        return samples


class TestTrainingPerformance:
    """Performance tests for model training."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = TrainingService()
        self.benchmark = PerformanceBenchmark()
    
    @patch('automl_framework.services.training_service.torch')
    def test_training_setup_performance(self, mock_torch):
        """Test training setup performance."""
        # Mock architecture
        architecture = MockArchitectureGenerator.create_simple_mlp()
        
        # Mock training data
        X_train = np.random.random((10000, 784))
        y_train = np.random.randint(0, 10, 10000)
        X_val = np.random.random((2000, 784))
        y_val = np.random.randint(0, 10, 2000)
        
        # Benchmark training setup
        setup_result = self.benchmark.benchmark_function(
            self._setup_training,
            architecture,
            X_train, y_train, X_val, y_val
        )
        
        # Performance assertions
        assert setup_result['execution_time'] < 10.0  # Setup should be fast
        assert setup_result['peak_memory_mb'] < 1000
        
        # Verify setup worked
        setup_data = setup_result['result']
        assert 'model' in setup_data
        assert 'optimizer' in setup_data
        assert 'data_loaders' in setup_data
    
    def _setup_training(self, architecture, X_train, y_train, X_val, y_val):
        """Helper method to setup training."""
        # Mock training setup
        setup_data = {
            'model': Mock(),
            'optimizer': Mock(),
            'data_loaders': {
                'train': Mock(),
                'val': Mock()
            },
            'criterion': Mock()
        }
        
        # Simulate some setup time
        time.sleep(0.1)
        
        return setup_data
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        batch_sizes = [16, 32, 64, 128, 256]
        performance_results = {}
        
        for batch_size in batch_sizes:
            # Create mock batch data
            batch_data = np.random.random((batch_size, 784))
            batch_labels = np.random.randint(0, 10, batch_size)
            
            # Benchmark batch processing
            result = self.benchmark.benchmark_function(
                self._process_batch,
                batch_data, batch_labels
            )
            
            performance_results[batch_size] = {
                'execution_time': result['execution_time'],
                'peak_memory_mb': result['peak_memory_mb']
            }
            
            # Performance should scale with batch size
            assert result['execution_time'] < batch_size * 0.001
        
        # Verify performance scaling
        times = [performance_results[bs]['execution_time'] for bs in batch_sizes]
        
        # Time should generally increase with batch size (but not exponentially)
        for i in range(1, len(times)):
            if times[i] > times[i-1]:  # Allow for some variation
                growth_factor = times[i] / times[i-1]
                batch_factor = batch_sizes[i] / batch_sizes[i-1]
                assert growth_factor <= batch_factor * 2
    
    def _process_batch(self, batch_data, batch_labels):
        """Helper method to process a batch."""
        # Mock batch processing
        processed_data = batch_data * 2  # Some computation
        predictions = np.random.random((len(batch_data), 10))
        loss = np.random.random()
        
        return {
            'predictions': predictions,
            'loss': loss,
            'processed_data': processed_data
        }


class TestEvaluationPerformance:
    """Performance tests for model evaluation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = EvaluationService()
        self.benchmark = PerformanceBenchmark()
    
    def test_metrics_calculation_performance(self):
        """Test performance of metrics calculation."""
        dataset_sizes = [1000, 5000, 10000, 50000]
        performance_results = {}
        
        for size in dataset_sizes:
            # Generate mock predictions and labels
            y_true = np.random.randint(0, 10, size)
            y_pred = np.random.randint(0, 10, size)
            y_prob = np.random.random((size, 10))
            
            # Benchmark metrics calculation
            result = self.benchmark.benchmark_function(
                self._calculate_comprehensive_metrics,
                y_true, y_pred, y_prob
            )
            
            performance_results[size] = {
                'execution_time': result['execution_time'],
                'peak_memory_mb': result['peak_memory_mb']
            }
            
            # Performance assertions
            assert result['execution_time'] < size * 0.0001  # Should be very fast
            assert result['peak_memory_mb'] < size * 0.001
        
        # Verify linear scaling
        times = [performance_results[size]['execution_time'] for size in dataset_sizes]
        
        for i in range(1, len(times)):
            growth_factor = times[i] / times[i-1] if times[i-1] > 0 else 1
            size_factor = dataset_sizes[i] / dataset_sizes[i-1]
            assert growth_factor <= size_factor * 1.5  # Allow some overhead
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_prob):
        """Helper method to calculate comprehensive metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_auc_score
        )
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred)
        
        # ROC AUC (for multiclass)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            metrics['roc_auc'] = None
        
        return metrics
    
    def test_model_comparison_performance(self):
        """Test performance of model comparison."""
        n_models = 10
        n_samples = 5000
        
        # Generate mock results for multiple models
        model_results = []
        for i in range(n_models):
            y_true = np.random.randint(0, 5, n_samples)
            y_pred = np.random.randint(0, 5, n_samples)
            y_prob = np.random.random((n_samples, 5))
            
            model_results.append({
                'model_id': f'model_{i}',
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            })
        
        # Benchmark model comparison
        comparison_result = self.benchmark.benchmark_function(
            self._compare_models,
            model_results
        )
        
        # Performance assertions
        assert comparison_result['execution_time'] < 30.0  # Should complete reasonably fast
        assert comparison_result['peak_memory_mb'] < 1000
        
        # Verify comparison worked
        comparison_data = comparison_result['result']
        assert 'model_rankings' in comparison_data
        assert 'statistical_tests' in comparison_data
        assert len(comparison_data['model_rankings']) == n_models
    
    def _compare_models(self, model_results):
        """Helper method to compare multiple models."""
        from sklearn.metrics import accuracy_score
        
        model_rankings = []
        
        for result in model_results:
            accuracy = accuracy_score(result['y_true'], result['y_pred'])
            model_rankings.append({
                'model_id': result['model_id'],
                'accuracy': accuracy,
                'rank': 0  # Will be filled later
            })
        
        # Sort by accuracy
        model_rankings.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Assign ranks
        for i, model in enumerate(model_rankings):
            model['rank'] = i + 1
        
        # Mock statistical tests
        statistical_tests = {
            'friedman_test': {'statistic': 15.2, 'p_value': 0.001},
            'nemenyi_test': {'critical_difference': 2.5}
        }
        
        return {
            'model_rankings': model_rankings,
            'statistical_tests': statistical_tests
        }


class TestRegressionTests:
    """Regression tests to ensure performance doesn't degrade."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = PerformanceBenchmark()
        
        # Performance baselines (in seconds)
        self.baselines = {
            'data_analysis_1k': 0.5,
            'data_analysis_10k': 2.0,
            'preprocessing_5k': 10.0,
            'feature_engineering_5k': 30.0,
            'architecture_encoding_100': 1.0,
            'hyperparameter_sampling_1k': 2.0,
            'metrics_calculation_10k': 1.0,
            'model_comparison_10': 5.0
        }
    
    def test_data_processing_regression(self, test_data_manager):
        """Test data processing performance regression."""
        # Test 1k samples
        df, temp_file = test_data_manager.create_tabular_dataset(n_samples=1000)
        
        service = DataProcessingService()
        result = self.benchmark.benchmark_function(
            service._analyze_dataset_internal,
            temp_file
        )
        
        self.benchmark.assert_performance_regression(
            result['execution_time'],
            self.baselines['data_analysis_1k'],
            tolerance=0.5  # Allow 50% regression
        )
        
        # Test 10k samples
        df, temp_file = test_data_manager.create_tabular_dataset(n_samples=10000)
        
        result = self.benchmark.benchmark_function(
            service._analyze_dataset_internal,
            temp_file
        )
        
        self.benchmark.assert_performance_regression(
            result['execution_time'],
            self.baselines['data_analysis_10k'],
            tolerance=0.5
        )
    
    def test_architecture_processing_regression(self):
        """Test architecture processing performance regression."""
        # Create 100 architectures
        architectures = []
        for i in range(100):
            arch = MockArchitectureGenerator.create_simple_mlp()
            architectures.append(arch)
        
        # Test encoding performance
        result = self.benchmark.benchmark_function(
            self._encode_architectures_simple,
            architectures
        )
        
        self.benchmark.assert_performance_regression(
            result['execution_time'],
            self.baselines['architecture_encoding_100'],
            tolerance=0.3
        )
    
    def _encode_architectures_simple(self, architectures):
        """Simple architecture encoding for regression test."""
        encoded = []
        for arch in architectures:
            encoded.append({
                'id': arch.id,
                'layers': len(arch.layers),
                'params': arch.parameter_count
            })
        return encoded
    
    def test_metrics_calculation_regression(self):
        """Test metrics calculation performance regression."""
        # Generate 10k samples
        y_true = np.random.randint(0, 10, 10000)
        y_pred = np.random.randint(0, 10, 10000)
        
        result = self.benchmark.benchmark_function(
            self._calculate_basic_metrics,
            y_true, y_pred
        )
        
        self.benchmark.assert_performance_regression(
            result['execution_time'],
            self.baselines['metrics_calculation_10k'],
            tolerance=0.3
        )
    
    def _calculate_basic_metrics(self, y_true, y_pred):
        """Basic metrics calculation for regression test."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }


if __name__ == "__main__":
    pytest.main([__file__])