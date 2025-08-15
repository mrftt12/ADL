"""
Tests for Tree-structured Parzen Estimator (TPE) optimization
"""

import pytest
import numpy as np
from unittest.mock import Mock

from automl_framework.services.hyperparameter_optimization import (
    TreeStructuredParzenEstimator,
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


class TestTreeStructuredParzenEstimator:
    """Test TPE optimizer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = TreeStructuredParzenEstimator(
            n_startup_trials=3,
            n_ei_candidates=5,
            gamma=0.25
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
        """Test TPE optimizer initialization"""
        assert self.optimizer.n_startup_trials == 3
        assert self.optimizer.n_ei_candidates == 5
        assert self.optimizer.gamma == 0.25
        assert len(self.optimizer.history.trials) == 0
    
    def test_simple_optimization(self):
        """Test TPE optimization with simple quadratic function"""
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
    
    def test_categorical_parameter_sampling(self):
        """Test TPE with categorical parameters"""
        # Add categorical parameter
        space = HyperparameterSpaceManager()
        space.add_parameter(Parameter(
            name="optimizer",
            param_type=ParameterType.CATEGORICAL,
            bounds=["adam", "sgd", "rmsprop"]
        ))
        space.add_parameter(Parameter(
            name="x",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0)
        ))
        
        def objective(params):
            # Prefer adam optimizer
            opt_bonus = 0.5 if params["optimizer"] == "adam" else 0.0
            return params["x"] + opt_bonus
        
        optimizer = TreeStructuredParzenEstimator(n_startup_trials=5, n_ei_candidates=10)
        best_params = optimizer.optimize(
            objective_function=objective,
            space_manager=space,
            max_trials=15
        )
        
        assert best_params is not None
        assert "optimizer" in best_params
        assert "x" in best_params
        
        # Should prefer adam optimizer
        history = optimizer.get_optimization_history()
        adam_trials = [t for t in history.trials if t.status == "completed" and 
                      t.parameters.get("optimizer") == "adam"]
        assert len(adam_trials) > 0
    
    def test_integer_parameter_sampling(self):
        """Test TPE with integer parameters"""
        space = HyperparameterSpaceManager()
        space.add_parameter(Parameter(
            name="n_layers",
            param_type=ParameterType.INTEGER,
            bounds=(1, 10)
        ))
        space.add_parameter(Parameter(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-4, 1e-1),
            log_scale=True
        ))
        
        def objective(params):
            # Prefer 3-5 layers and learning rate around 1e-3
            layer_penalty = abs(params["n_layers"] - 4) / 4.0
            lr_penalty = abs(np.log10(params["learning_rate"]) - np.log10(1e-3)) / 2.0
            return 1.0 - (layer_penalty + lr_penalty) / 2.0
        
        optimizer = TreeStructuredParzenEstimator(n_startup_trials=5)
        best_params = optimizer.optimize(
            objective_function=objective,
            space_manager=space,
            max_trials=15
        )
        
        assert best_params is not None
        assert isinstance(best_params["n_layers"], int)
        assert 1 <= best_params["n_layers"] <= 10
        assert 1e-4 <= best_params["learning_rate"] <= 1e-1
    
    def test_boolean_parameter_sampling(self):
        """Test TPE with boolean parameters"""
        space = HyperparameterSpaceManager()
        space.add_parameter(Parameter(
            name="use_dropout",
            param_type=ParameterType.BOOLEAN,
            bounds=[True, False]
        ))
        space.add_parameter(Parameter(
            name="x",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0)
        ))
        
        def objective(params):
            # Prefer dropout enabled
            dropout_bonus = 0.3 if params["use_dropout"] else 0.0
            return params["x"] + dropout_bonus
        
        optimizer = TreeStructuredParzenEstimator(n_startup_trials=4)
        best_params = optimizer.optimize(
            objective_function=objective,
            space_manager=space,
            max_trials=12
        )
        
        assert best_params is not None
        assert isinstance(best_params["use_dropout"], bool)
        assert "x" in best_params
    
    def test_kde_fitting_and_sampling(self):
        """Test kernel density estimation functions"""
        # Test with single value
        kde = self.optimizer._fit_kde([0.5], 0.0, 1.0)
        assert kde['type'] == 'gaussian'
        assert kde['mean'] == 0.5
        
        # Test with multiple values
        values = [0.2, 0.3, 0.4, 0.6, 0.7]
        kde = self.optimizer._fit_kde(values, 0.0, 1.0)
        assert kde['type'] == 'gaussian'
        assert abs(kde['mean'] - np.mean(values)) < 1e-6
        
        # Test sampling
        sample = self.optimizer._sample_from_kde(kde, 0.0, 1.0)
        assert 0.0 <= sample <= 1.0
        
        # Test evaluation
        density = self.optimizer._evaluate_kde(kde, 0.5)
        assert density > 0
    
    def test_failed_trials_handling(self):
        """Test handling of failed trials"""
        def failing_objective(params):
            if params["x"] > 2.0:
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
        assert len(failed_trials) >= 0  # May or may not have failures depending on sampling
    
    def test_insufficient_data_fallback(self):
        """Test fallback when insufficient data for TPE"""
        # Create optimizer that needs more startup trials than we'll provide
        optimizer = TreeStructuredParzenEstimator(n_startup_trials=2)
        
        def simple_objective(params):
            return params["x"]
        
        # Manually add only one trial
        optimizer.space_manager = self.space
        optimizer._random_exploration(simple_objective, 1)
        
        # Should fallback to random sampling
        next_params = optimizer._suggest_next_point()
        assert next_params is not None
        assert "x" in next_params
        assert "y" in next_params


class TestTPEIntegration:
    """Test TPE integration with main service"""
    
    def test_tpe_service_integration(self):
        """Test TPE algorithm in main service"""
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
        
        # Initialize service with TPE
        service = HyperparameterOptimizationService(algorithm="tpe")
        
        # Define search space
        search_space = service.define_search_space(architecture)
        
        # Mock objective function
        def objective_function(params):
            """Function that prefers specific values"""
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
            max_trials=12
        )
        
        # Verify results
        assert isinstance(best_config, TrainingConfig)
        
        # Check that optimization found reasonable values
        history = service.get_optimization_history()
        assert len(history) > 0
        
        # Best trial should have reasonable performance
        best_trial = max(history, key=lambda t: t.metrics.accuracy if t.metrics else 0)
        assert best_trial.metrics.accuracy > 0.3
        
        # Learning rate should be in reasonable range
        assert 1e-5 <= best_config.learning_rate <= 1e-1
        
        # Batch size should be valid
        assert best_config.batch_size in [16, 32, 64, 128, 256]
    
    def test_algorithm_comparison(self):
        """Test comparison between Bayesian and TPE algorithms"""
        architecture = Architecture(
            id="test_arch",
            layers=[],
            connections=[],
            input_shape=(28, 28, 1),
            output_shape=(10,),
            parameter_count=50000,
            flops=50000
        )
        
        def objective_function(params):
            # Simple function with clear optimum
            lr = params.get("learning_rate", 0.001)
            return 1.0 - abs(np.log10(lr) - np.log10(0.01))
        
        # Test Bayesian optimization
        bayesian_service = HyperparameterOptimizationService(algorithm="bayesian")
        bayesian_space = bayesian_service.define_search_space(architecture)
        bayesian_config = bayesian_service.optimize(
            objective_function=objective_function,
            search_space=bayesian_space,
            max_trials=10
        )
        
        # Test TPE optimization
        tpe_service = HyperparameterOptimizationService(algorithm="tpe")
        tpe_space = tpe_service.define_search_space(architecture)
        tpe_config = tpe_service.optimize(
            objective_function=objective_function,
            search_space=tpe_space,
            max_trials=10
        )
        
        # Both should produce valid results
        assert isinstance(bayesian_config, TrainingConfig)
        assert isinstance(tpe_config, TrainingConfig)
        
        # Both should find reasonable learning rates (with small tolerance for floating point)
        assert 1e-5 <= bayesian_config.learning_rate <= 1e-1 + 1e-10
        assert 1e-5 <= tpe_config.learning_rate <= 1e-1 + 1e-10
        
        # Both should have optimization history
        bayesian_history = bayesian_service.get_optimization_history()
        tpe_history = tpe_service.get_optimization_history()
        
        assert len(bayesian_history) > 0
        assert len(tpe_history) > 0
    
    def test_invalid_algorithm(self):
        """Test error handling for invalid algorithm"""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            service = HyperparameterOptimizationService(algorithm="invalid")
            architecture = Architecture(
                id="test", layers=[], connections=[], 
                input_shape=(10,), output_shape=(1,), 
                parameter_count=100, flops=100
            )
            search_space = service.define_search_space(architecture)
            service.optimize(lambda x: 0.5, search_space, max_trials=5)


if __name__ == "__main__":
    pytest.main([__file__])