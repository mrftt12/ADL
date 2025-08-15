"""
Tests for Bayesian optimization functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from automl_framework.services.hyperparameter_optimization import (
    BayesianOptimizer,
    AcquisitionFunction,
    OptimizationHistory,
    HyperparameterOptimizationService,
    HyperparameterSpaceManager,
    Parameter,
    ParameterType,
    create_default_hyperparameter_space
)
from automl_framework.core.interfaces import (
    Architecture,
    Trial,
    PerformanceMetrics,
    TrainingConfig
)


class TestOptimizationHistory:
    """Test OptimizationHistory class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.history = OptimizationHistory()
    
    def test_add_trial(self):
        """Test adding trials to history"""
        metrics = PerformanceMetrics(
            accuracy=0.85,
            loss=0.15,
            precision=0.8,
            recall=0.9,
            f1_score=0.85,
            training_time=100.0,
            inference_time=0.1,
            additional_metrics={}
        )
        
        trial = Trial(
            id="trial_1",
            parameters={"learning_rate": 0.001, "batch_size": 32},
            metrics=metrics,
            status="completed",
            duration=100.0
        )
        
        self.history.add_trial(trial)
        
        assert len(self.history.trials) == 1
        assert self.history.best_trial == trial
        assert len(self.history.convergence_history) == 1
        assert self.history.convergence_history[0] == 0.85
    
    def test_update_best_trial(self):
        """Test updating best trial"""
        # Add first trial
        trial1 = Trial(
            id="trial_1",
            parameters={"learning_rate": 0.001},
            metrics=PerformanceMetrics(
                accuracy=0.8, loss=0.2, precision=0.8, recall=0.8, f1_score=0.8,
                training_time=100.0, inference_time=0.1, additional_metrics={}
            ),
            status="completed",
            duration=100.0
        )
        self.history.add_trial(trial1)
        
        # Add better trial
        trial2 = Trial(
            id="trial_2",
            parameters={"learning_rate": 0.01},
            metrics=PerformanceMetrics(
                accuracy=0.9, loss=0.1, precision=0.9, recall=0.9, f1_score=0.9,
                training_time=100.0, inference_time=0.1, additional_metrics={}
            ),
            status="completed",
            duration=100.0
        )
        self.history.add_trial(trial2)
        
        assert self.history.best_trial == trial2
        assert self.history.best_trial.metrics.accuracy == 0.9
    
    def test_convergence_detection(self):
        """Test convergence detection"""
        # Add trials with no improvement
        np.random.seed(42)  # For reproducible test
        for i in range(15):
            trial = Trial(
                id=f"trial_{i}",
                parameters={"learning_rate": 0.001},
                metrics=PerformanceMetrics(
                    accuracy=0.8 + np.random.normal(0, 0.00001),  # Very small noise
                    loss=0.2, precision=0.8, recall=0.8, f1_score=0.8,
                    training_time=100.0, inference_time=0.1, additional_metrics={}
                ),
                status="completed",
                duration=100.0
            )
            self.history.add_trial(trial)
        
        # Should converge with default patience=10 and small improvement threshold
        assert self.history.has_converged(patience=10, min_improvement=1e-5)
        
        # Should not converge with higher improvement threshold
        assert not self.history.has_converged(patience=5, min_improvement=1e-3)


class TestBayesianOptimizer:
    """Test BayesianOptimizer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = BayesianOptimizer(
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            n_initial_points=3
        )
        
        # Create simple hyperparameter space
        self.space = HyperparameterSpaceManager()
        self.space.add_parameter(Parameter(
            name="x",
            param_type=ParameterType.CONTINUOUS,
            bounds=(-5.0, 5.0)
        ))
        self.space.add_parameter(Parameter(
            name="y",
            param_type=ParameterType.CONTINUOUS,
            bounds=(-5.0, 5.0)
        ))
    
    def test_initialization(self):
        """Test optimizer initialization"""
        assert self.optimizer.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT
        assert self.optimizer.n_initial_points == 3
        assert self.optimizer.gp_model is None
        assert len(self.optimizer.history.trials) == 0
    
    def test_simple_optimization(self):
        """Test optimization with simple quadratic function"""
        def objective(params):
            """Simple quadratic function with optimum at (1, 2)"""
            x, y = params["x"], params["y"]
            return 1.0 - ((x - 1)**2 + (y - 2)**2) / 10.0
        
        # Run optimization
        best_params = self.optimizer.optimize(
            objective_function=objective,
            space_manager=self.space,
            max_trials=10
        )
        
        # Check that optimization ran
        assert len(self.optimizer.history.trials) > 0
        assert best_params is not None
        assert "x" in best_params
        assert "y" in best_params
        
        # Check that best parameters are reasonable
        best_score = objective(best_params)
        assert best_score > 0.5  # Should find reasonably good solution
    
    def test_acquisition_functions(self):
        """Test different acquisition functions"""
        def simple_objective(params):
            return 1.0 - params["x"]**2
        
        # Test Expected Improvement
        optimizer_ei = BayesianOptimizer(
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            n_initial_points=2
        )
        
        best_params_ei = optimizer_ei.optimize(
            objective_function=simple_objective,
            space_manager=self.space,
            max_trials=5
        )
        
        assert best_params_ei is not None
        
        # Test Upper Confidence Bound
        optimizer_ucb = BayesianOptimizer(
            acquisition_function=AcquisitionFunction.UPPER_CONFIDENCE_BOUND,
            n_initial_points=2
        )
        
        best_params_ucb = optimizer_ucb.optimize(
            objective_function=simple_objective,
            space_manager=self.space,
            max_trials=5
        )
        
        assert best_params_ucb is not None
    
    def test_failed_trials(self):
        """Test handling of failed trials"""
        def failing_objective(params):
            if params["x"] > 0:
                raise ValueError("Simulated failure")
            return 1.0 - params["x"]**2
        
        # Should handle failures gracefully
        best_params = self.optimizer.optimize(
            objective_function=failing_objective,
            space_manager=self.space,
            max_trials=8
        )
        
        # Should still return some result
        assert best_params is not None
        
        # Check that some trials failed
        failed_trials = [t for t in self.optimizer.history.trials if t.status == "failed"]
        assert len(failed_trials) > 0
    
    @patch('automl_framework.services.hyperparameter_optimization.minimize')
    def test_acquisition_optimization_fallback(self, mock_minimize):
        """Test fallback when acquisition optimization fails"""
        # Mock minimize to fail
        mock_minimize.return_value.success = False
        
        def simple_objective(params):
            return 1.0 - params["x"]**2
        
        # Set space manager first
        self.optimizer.space_manager = self.space
        
        # Run a few trials to initialize GP
        self.optimizer._random_exploration(simple_objective, 3)
        
        # Should fallback to random sampling
        next_params = self.optimizer._suggest_next_point()
        assert next_params is not None
        assert "x" in next_params
        assert "y" in next_params


class TestHyperparameterOptimizationService:
    """Test HyperparameterOptimizationService class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = HyperparameterOptimizationService()
        
        # Create mock architecture
        self.architecture = Architecture(
            id="test_arch",
            layers=[],
            connections=[],
            input_shape=(224, 224, 3),
            output_shape=(10,),
            parameter_count=1000000,
            flops=1000000
        )
    
    def test_define_search_space(self):
        """Test search space definition"""
        search_space = self.service.define_search_space(self.architecture)
        
        assert search_space is not None
        assert len(search_space.parameters) > 0
        assert len(search_space.parameter_types) > 0
        
        # Check that common parameters are present
        assert "learning_rate" in search_space.parameters
        assert "batch_size" in search_space.parameters
        assert "optimizer" in search_space.parameters
    
    def test_optimization_workflow(self):
        """Test complete optimization workflow"""
        # Define search space
        search_space = self.service.define_search_space(self.architecture)
        
        # Mock objective function
        def mock_objective(params):
            # Simple function that prefers smaller learning rates
            lr = params.get("learning_rate", 0.001)
            return 1.0 - abs(lr - 0.001) * 1000
        
        # Run optimization
        best_config = self.service.optimize(
            objective_function=mock_objective,
            search_space=search_space,
            max_trials=8
        )
        
        # Check result
        assert isinstance(best_config, TrainingConfig)
        assert hasattr(best_config, 'learning_rate')
        assert hasattr(best_config, 'batch_size')
        assert hasattr(best_config, 'optimizer')
        assert hasattr(best_config, 'epochs')
        
        # Check that optimization history is available
        history = self.service.get_optimization_history()
        assert len(history) > 0
    
    def test_suggest_next_parameters(self):
        """Test parameter suggestion"""
        # Need to run optimization first
        search_space = self.service.define_search_space(self.architecture)
        
        def mock_objective(params):
            return np.random.random()
        
        # Run a few trials
        self.service.optimize(
            objective_function=mock_objective,
            search_space=search_space,
            max_trials=5
        )
        
        # Should be able to suggest next parameters
        next_params = self.service.suggest_next_parameters()
        assert next_params is not None
        assert isinstance(next_params, dict)
    
    def test_update_trial_result(self):
        """Test updating trial results"""
        # Define search space first
        search_space = self.service.define_search_space(self.architecture)
        
        # Update trial result
        params = {"learning_rate": 0.001, "batch_size": 32}
        metrics = PerformanceMetrics(
            accuracy=0.85,
            loss=0.15,
            precision=0.8,
            recall=0.9,
            f1_score=0.85,
            training_time=100.0,
            inference_time=0.1,
            additional_metrics={}
        )
        
        self.service.update_trial_result(params, metrics)
        
        # Check that trial was added
        history = self.service.get_optimization_history()
        assert len(history) == 1
        assert history[0].parameters == params
        assert history[0].metrics.accuracy == 0.85


class TestIntegration:
    """Integration tests for hyperparameter optimization"""
    
    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization"""
        # Create architecture
        architecture = Architecture(
            id="test_arch",
            layers=[],
            connections=[],
            input_shape=(32, 32, 3),
            output_shape=(10,),
            parameter_count=100000,
            flops=100000
        )
        
        # Initialize service
        service = HyperparameterOptimizationService()
        
        # Define search space
        search_space = service.define_search_space(architecture)
        
        # Mock objective function that has a clear optimum
        def objective_function(params):
            """Function with optimum at lr=0.01, batch_size=64"""
            lr = params.get("learning_rate", 0.001)
            batch_size = params.get("batch_size", 32)
            
            # Penalty for being far from optimal values
            lr_penalty = abs(np.log10(lr) - np.log10(0.01))
            batch_penalty = abs(batch_size - 64) / 64.0
            
            return 1.0 - (lr_penalty + batch_penalty) / 2.0
        
        # Run optimization
        best_config = service.optimize(
            objective_function=objective_function,
            search_space=search_space,
            max_trials=15
        )
        
        # Verify results
        assert isinstance(best_config, TrainingConfig)
        
        # Check that optimization found reasonable values
        history = service.get_optimization_history()
        assert len(history) > 0
        
        # Best trial should have reasonable performance
        best_trial = max(history, key=lambda t: t.metrics.accuracy if t.metrics else 0)
        assert best_trial.metrics.accuracy > 0.3  # More lenient threshold
        
        # Learning rate should be in reasonable range
        assert 1e-5 <= best_config.learning_rate <= 1e-1
        
        # Batch size should be valid
        assert best_config.batch_size in [16, 32, 64, 128, 256]


if __name__ == "__main__":
    pytest.main([__file__])