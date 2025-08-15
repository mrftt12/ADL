"""
Hyperparameter optimization service for AutoML Framework
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import random
from scipy import stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

from ..core.interfaces import (
    IHyperparameterOptimizer, 
    HyperparameterSpace, 
    Architecture, 
    TrainingConfig, 
    Trial,
    PerformanceMetrics
)
from ..core.exceptions import AutoMLException


logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of hyperparameters"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    INTEGER = "integer"
    BOOLEAN = "boolean"


@dataclass
class Parameter:
    """Individual hyperparameter definition"""
    name: str
    param_type: ParameterType
    bounds: Union[Tuple[float, float], List[Any]]
    log_scale: bool = False
    default: Optional[Any] = None
    constraints: List[str] = field(default_factory=list)


@dataclass
class ParameterConstraint:
    """Constraint between parameters"""
    constraint_type: str  # 'conditional', 'dependency', 'mutual_exclusive'
    parameters: List[str]
    condition: str
    action: str


class HyperparameterSpaceManager:
    """Manages hyperparameter space definition and operations"""
    
    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}
        self.constraints: List[ParameterConstraint] = []
        self.parameter_dependencies: Dict[str, List[str]] = {}
    
    def add_parameter(self, parameter: Parameter) -> None:
        """Add a parameter to the search space"""
        self.parameters[parameter.name] = parameter
        logger.debug(f"Added parameter: {parameter.name} of type {parameter.param_type}")
    
    def add_constraint(self, constraint: ParameterConstraint) -> None:
        """Add a constraint between parameters"""
        self.constraints.append(constraint)
        
        # Update parameter dependencies
        for param in constraint.parameters:
            if param not in self.parameter_dependencies:
                self.parameter_dependencies[param] = []
            for other_param in constraint.parameters:
                if other_param != param and other_param not in self.parameter_dependencies[param]:
                    self.parameter_dependencies[param].append(other_param)
        
        logger.debug(f"Added constraint: {constraint.constraint_type} for parameters {constraint.parameters}")
    
    def sample_parameters(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        """Sample parameters from the search space"""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            # Sample each parameter
            for param_name, param in self.parameters.items():
                sample[param_name] = self._sample_single_parameter(param)
            
            # Apply constraints
            sample = self._apply_constraints(sample)
            samples.append(sample)
        
        return samples
    
    def _sample_single_parameter(self, param: Parameter) -> Any:
        """Sample a single parameter value"""
        if param.param_type == ParameterType.CONTINUOUS:
            low, high = param.bounds
            if param.log_scale:
                log_low, log_high = np.log(low), np.log(high)
                value = np.exp(np.random.uniform(log_low, log_high))
            else:
                value = np.random.uniform(low, high)
            return float(value)
        
        elif param.param_type == ParameterType.INTEGER:
            low, high = param.bounds
            if param.log_scale:
                log_low, log_high = np.log(max(1, low)), np.log(high)
                value = int(np.exp(np.random.uniform(log_low, log_high)))
            else:
                value = np.random.randint(low, high + 1)
            return int(value)
        
        elif param.param_type == ParameterType.DISCRETE:
            return random.choice(param.bounds)
        
        elif param.param_type == ParameterType.CATEGORICAL:
            return random.choice(param.bounds)
        
        elif param.param_type == ParameterType.BOOLEAN:
            return random.choice(param.bounds if param.bounds else [True, False])
        
        else:
            raise ValueError(f"Unknown parameter type: {param.param_type}")
    
    def _apply_constraints(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply constraints to a parameter sample"""
        for constraint in self.constraints:
            if constraint.constraint_type == "conditional":
                sample = self._apply_conditional_constraint(sample, constraint)
            elif constraint.constraint_type == "dependency":
                sample = self._apply_dependency_constraint(sample, constraint)
            elif constraint.constraint_type == "mutual_exclusive":
                sample = self._apply_mutual_exclusive_constraint(sample, constraint)
        
        return sample
    
    def _apply_conditional_constraint(self, sample: Dict[str, Any], constraint: ParameterConstraint) -> Dict[str, Any]:
        """Apply conditional constraint (if param1 == value, then param2 = specific_value)"""
        # Parse condition: "learning_rate > 0.01"
        condition_parts = constraint.condition.split()
        if len(condition_parts) == 3:
            param_name, operator, value = condition_parts
            param_value = sample.get(param_name)
            
            # Evaluate condition
            condition_met = False
            try:
                value = float(value) if '.' in value else int(value)
                if operator == '>':
                    condition_met = param_value > value
                elif operator == '<':
                    condition_met = param_value < value
                elif operator == '==':
                    condition_met = param_value == value
                elif operator == '!=':
                    condition_met = param_value != value
            except (ValueError, TypeError):
                # Handle string comparisons
                if operator == '==':
                    condition_met = str(param_value) == str(value)
                elif operator == '!=':
                    condition_met = str(param_value) != str(value)
            
            # Apply action if condition is met
            if condition_met:
                action_parts = constraint.action.split('=')
                if len(action_parts) == 2:
                    target_param, target_value = action_parts
                    target_param = target_param.strip()
                    target_value = target_value.strip()
                    
                    # Convert target value to appropriate type
                    if target_param in self.parameters:
                        param_def = self.parameters[target_param]
                        if param_def.param_type == ParameterType.BOOLEAN:
                            sample[target_param] = target_value.lower() == 'true'
                        elif param_def.param_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                            sample[target_param] = float(target_value) if param_def.param_type == ParameterType.CONTINUOUS else int(target_value)
                        else:
                            sample[target_param] = target_value
        
        return sample
    
    def _apply_dependency_constraint(self, sample: Dict[str, Any], constraint: ParameterConstraint) -> Dict[str, Any]:
        """Apply dependency constraint (param2 depends on param1)"""
        # Implementation for parameter dependencies
        return sample
    
    def _apply_mutual_exclusive_constraint(self, sample: Dict[str, Any], constraint: ParameterConstraint) -> Dict[str, Any]:
        """Apply mutual exclusive constraint (only one of the parameters can be active)"""
        # Implementation for mutual exclusivity
        return sample
    
    def encode_parameters(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Encode parameters to numerical array for optimization"""
        encoded = []
        
        for param_name in sorted(self.parameters.keys()):
            param = self.parameters[param_name]
            value = parameters.get(param_name)
            
            if param.param_type == ParameterType.CONTINUOUS:
                if param.log_scale:
                    encoded_value = np.log(value)
                else:
                    encoded_value = value
                # Normalize to [0, 1]
                low, high = param.bounds
                if param.log_scale:
                    low, high = np.log(low), np.log(high)
                encoded_value = (encoded_value - low) / (high - low)
                encoded.append(encoded_value)
            
            elif param.param_type == ParameterType.INTEGER:
                if param.log_scale:
                    encoded_value = np.log(value)
                    low, high = np.log(max(1, param.bounds[0])), np.log(param.bounds[1])
                else:
                    encoded_value = value
                    low, high = param.bounds
                encoded_value = (encoded_value - low) / (high - low)
                encoded.append(encoded_value)
            
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.DISCRETE]:
                # One-hot encoding
                categories = param.bounds
                one_hot = [0.0] * len(categories)
                if value in categories:
                    one_hot[categories.index(value)] = 1.0
                encoded.extend(one_hot)
            
            elif param.param_type == ParameterType.BOOLEAN:
                encoded.append(1.0 if value else 0.0)
        
        return np.array(encoded)
    
    def decode_parameters(self, encoded: np.ndarray) -> Dict[str, Any]:
        """Decode numerical array back to parameters"""
        decoded = {}
        idx = 0
        
        for param_name in sorted(self.parameters.keys()):
            param = self.parameters[param_name]
            
            if param.param_type == ParameterType.CONTINUOUS:
                # Denormalize from [0, 1]
                normalized_value = encoded[idx]
                low, high = param.bounds
                if param.log_scale:
                    low, high = np.log(low), np.log(high)
                    value = np.exp(normalized_value * (high - low) + low)
                else:
                    value = normalized_value * (high - low) + low
                decoded[param_name] = float(value)
                idx += 1
            
            elif param.param_type == ParameterType.INTEGER:
                normalized_value = encoded[idx]
                low, high = param.bounds
                if param.log_scale:
                    low, high = np.log(max(1, low)), np.log(high)
                    value = int(np.exp(normalized_value * (high - low) + low))
                else:
                    value = int(normalized_value * (high - low) + low)
                decoded[param_name] = value
                idx += 1
            
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.DISCRETE]:
                categories = param.bounds
                one_hot = encoded[idx:idx + len(categories)]
                selected_idx = np.argmax(one_hot)
                decoded[param_name] = categories[selected_idx]
                idx += len(categories)
            
            elif param.param_type == ParameterType.BOOLEAN:
                decoded[param_name] = encoded[idx] > 0.5
                idx += 1
        
        return decoded
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for numerical optimization"""
        bounds = []
        
        for param_name in sorted(self.parameters.keys()):
            param = self.parameters[param_name]
            
            if param.param_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                bounds.append((0.0, 1.0))  # Normalized bounds
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.DISCRETE]:
                # One bound per category
                for _ in param.bounds:
                    bounds.append((0.0, 1.0))
            elif param.param_type == ParameterType.BOOLEAN:
                bounds.append((0.0, 1.0))
        
        return bounds
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameter values against constraints"""
        # Check parameter bounds
        for param_name, value in parameters.items():
            if param_name not in self.parameters:
                return False
            
            param = self.parameters[param_name]
            
            if param.param_type == ParameterType.CONTINUOUS:
                low, high = param.bounds
                if not (low <= value <= high):
                    return False
            elif param.param_type == ParameterType.INTEGER:
                low, high = param.bounds
                if not (low <= value <= high) or not isinstance(value, int):
                    return False
            elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.DISCRETE]:
                if value not in param.bounds:
                    return False
            elif param.param_type == ParameterType.BOOLEAN:
                if not isinstance(value, bool):
                    return False
        
        # Check constraints
        for constraint in self.constraints:
            if not self._validate_constraint(parameters, constraint):
                return False
        
        return True
    
    def _validate_constraint(self, parameters: Dict[str, Any], constraint: ParameterConstraint) -> bool:
        """Validate a single constraint"""
        # Implementation depends on constraint type
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hyperparameter space to dictionary"""
        return {
            'parameters': {
                name: {
                    'name': param.name,
                    'param_type': param.param_type.value,
                    'bounds': param.bounds,
                    'log_scale': param.log_scale,
                    'default': param.default,
                    'constraints': param.constraints
                }
                for name, param in self.parameters.items()
            },
            'constraints': [
                {
                    'constraint_type': constraint.constraint_type,
                    'parameters': constraint.parameters,
                    'condition': constraint.condition,
                    'action': constraint.action
                }
                for constraint in self.constraints
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperparameterSpaceManager':
        """Create hyperparameter space from dictionary"""
        space = cls()
        
        # Load parameters
        for name, param_data in data.get('parameters', {}).items():
            param = Parameter(
                name=param_data['name'],
                param_type=ParameterType(param_data['param_type']),
                bounds=param_data['bounds'],
                log_scale=param_data.get('log_scale', False),
                default=param_data.get('default'),
                constraints=param_data.get('constraints', [])
            )
            space.add_parameter(param)
        
        # Load constraints
        for constraint_data in data.get('constraints', []):
            constraint = ParameterConstraint(
                constraint_type=constraint_data['constraint_type'],
                parameters=constraint_data['parameters'],
                condition=constraint_data['condition'],
                action=constraint_data['action']
            )
            space.add_constraint(constraint)
        
        return space


def create_default_hyperparameter_space(architecture: Architecture) -> HyperparameterSpaceManager:
    """Create default hyperparameter space for a given architecture"""
    space = HyperparameterSpaceManager()
    
    # Learning rate
    space.add_parameter(Parameter(
        name="learning_rate",
        param_type=ParameterType.CONTINUOUS,
        bounds=(1e-5, 1e-1),
        log_scale=True,
        default=1e-3
    ))
    
    # Batch size
    space.add_parameter(Parameter(
        name="batch_size",
        param_type=ParameterType.DISCRETE,
        bounds=[16, 32, 64, 128, 256],
        default=32
    ))
    
    # Optimizer
    space.add_parameter(Parameter(
        name="optimizer",
        param_type=ParameterType.CATEGORICAL,
        bounds=["adam", "sgd", "rmsprop", "adamw"],
        default="adam"
    ))
    
    # Weight decay
    space.add_parameter(Parameter(
        name="weight_decay",
        param_type=ParameterType.CONTINUOUS,
        bounds=(1e-6, 1e-2),
        log_scale=True,
        default=1e-4
    ))
    
    # Dropout rate
    space.add_parameter(Parameter(
        name="dropout_rate",
        param_type=ParameterType.CONTINUOUS,
        bounds=(0.0, 0.8),
        default=0.1
    ))
    
    # Early stopping patience
    space.add_parameter(Parameter(
        name="early_stopping_patience",
        param_type=ParameterType.INTEGER,
        bounds=(5, 50),
        default=10
    ))
    
    # Epochs
    space.add_parameter(Parameter(
        name="epochs",
        param_type=ParameterType.INTEGER,
        bounds=(10, 200),
        default=100
    ))
    
    # Use mixed precision
    space.add_parameter(Parameter(
        name="use_mixed_precision",
        param_type=ParameterType.BOOLEAN,
        bounds=[True, False],
        default=False
    ))
    
    # Add constraints
    # If SGD optimizer, add momentum parameter
    space.add_constraint(ParameterConstraint(
        constraint_type="conditional",
        parameters=["optimizer", "momentum"],
        condition="optimizer == sgd",
        action="momentum = 0.9"
    ))
    
    return space


class AcquisitionFunction(Enum):
    """Types of acquisition functions for Bayesian optimization"""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"


@dataclass
class OptimizationHistory:
    """History of optimization trials"""
    trials: List[Trial] = field(default_factory=list)
    best_trial: Optional[Trial] = None
    convergence_history: List[float] = field(default_factory=list)
    
    def add_trial(self, trial: Trial) -> None:
        """Add a trial to the history"""
        self.trials.append(trial)
        self.convergence_history.append(trial.metrics.accuracy if trial.metrics else 0.0)
        
        # Update best trial
        if self.best_trial is None or (trial.metrics and trial.metrics.accuracy > self.best_trial.metrics.accuracy):
            self.best_trial = trial
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get parameters of the best trial"""
        return self.best_trial.parameters if self.best_trial else None
    
    def get_convergence_data(self) -> Tuple[List[float], List[float]]:
        """Get convergence data for plotting"""
        iterations = list(range(1, len(self.convergence_history) + 1))
        return iterations, self.convergence_history
    
    def has_converged(self, patience: int = 10, min_improvement: float = 1e-4) -> bool:
        """Check if optimization has converged"""
        if len(self.convergence_history) < patience:
            return False
        
        recent_scores = self.convergence_history[-patience:]
        improvement = max(recent_scores) - min(recent_scores)
        return improvement < min_improvement


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning"""
    
    def __init__(
        self,
        acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT,
        n_initial_points: int = 5,
        alpha: float = 1e-6,
        n_restarts_optimizer: int = 5,
        xi: float = 0.01,
        kappa: float = 2.576
    ):
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.xi = xi  # Exploration parameter for EI
        self.kappa = kappa  # Exploration parameter for UCB
        
        self.gp_model = None
        self.history = OptimizationHistory()
        self.space_manager = None
        
        logger.info(f"Initialized Bayesian optimizer with {acquisition_function.value} acquisition function")
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        space_manager: HyperparameterSpaceManager,
        max_trials: int = 50,
        convergence_patience: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization
        
        Args:
            objective_function: Function to optimize (should return higher values for better performance)
            space_manager: Hyperparameter space manager
            max_trials: Maximum number of trials
            convergence_patience: Number of trials to wait for convergence
            
        Returns:
            Best hyperparameter configuration
        """
        self.space_manager = space_manager
        self.history = OptimizationHistory()
        
        logger.info(f"Starting Bayesian optimization with {max_trials} max trials")
        
        # Initialize Gaussian Process model
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=self.alpha)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True
        )
        
        # Phase 1: Random exploration
        logger.info(f"Phase 1: Random exploration with {self.n_initial_points} points")
        self._random_exploration(objective_function, self.n_initial_points)
        
        # Phase 2: Bayesian optimization
        logger.info("Phase 2: Bayesian optimization")
        for trial_idx in range(self.n_initial_points, max_trials):
            if self.history.has_converged(patience=convergence_patience):
                logger.info(f"Converged after {trial_idx} trials")
                break
            
            # Suggest next point
            next_params = self._suggest_next_point()
            
            # Evaluate objective function
            try:
                score = objective_function(next_params)
                
                # Create trial
                trial = Trial(
                    id=f"trial_{trial_idx}",
                    parameters=next_params,
                    metrics=PerformanceMetrics(
                        accuracy=score,
                        loss=1.0 - score,  # Assuming score is accuracy
                        precision=0.0,
                        recall=0.0,
                        f1_score=0.0,
                        training_time=0.0,
                        inference_time=0.0,
                        additional_metrics={}
                    ),
                    status="completed",
                    duration=0.0
                )
                
                self.history.add_trial(trial)
                logger.info(f"Trial {trial_idx}: score={score:.4f}, best={self.history.best_trial.metrics.accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {trial_idx} failed: {e}")
                # Add failed trial
                trial = Trial(
                    id=f"trial_{trial_idx}",
                    parameters=next_params,
                    metrics=None,
                    status="failed",
                    duration=0.0
                )
                self.history.add_trial(trial)
        
        best_params = self.history.get_best_parameters()
        logger.info(f"Optimization completed. Best score: {self.history.best_trial.metrics.accuracy:.4f}")
        
        return best_params
    
    def _random_exploration(self, objective_function: Callable, n_points: int) -> None:
        """Perform random exploration to initialize the GP model"""
        if self.space_manager is None:
            raise ValueError("Space manager not set")
            
        for i in range(n_points):
            # Sample random parameters
            params = self.space_manager.sample_parameters(n_samples=1)[0]
            
            try:
                score = objective_function(params)
                
                trial = Trial(
                    id=f"random_{i}",
                    parameters=params,
                    metrics=PerformanceMetrics(
                        accuracy=score,
                        loss=1.0 - score,
                        precision=0.0,
                        recall=0.0,
                        f1_score=0.0,
                        training_time=0.0,
                        inference_time=0.0,
                        additional_metrics={}
                    ),
                    status="completed",
                    duration=0.0
                )
                
                self.history.add_trial(trial)
                logger.debug(f"Random exploration {i}: score={score:.4f}")
                
            except Exception as e:
                logger.error(f"Random exploration {i} failed: {e}")
    
    def _suggest_next_point(self) -> Dict[str, Any]:
        """Suggest next point to evaluate using acquisition function"""
        # Prepare training data for GP
        X_train = []
        y_train = []
        
        for trial in self.history.trials:
            if trial.status == "completed" and trial.metrics:
                encoded_params = self.space_manager.encode_parameters(trial.parameters)
                X_train.append(encoded_params)
                y_train.append(trial.metrics.accuracy)
        
        # Check if we have enough data
        if len(X_train) == 0:
            logger.warning("No completed trials found, falling back to random sampling")
            return self.space_manager.sample_parameters(n_samples=1)[0]
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Ensure X_train is 2D
        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)
        
        # Fit GP model
        self.gp_model.fit(X_train, y_train)
        
        # Optimize acquisition function
        bounds = self.space_manager.get_bounds()
        
        best_acquisition = -np.inf
        best_x = None
        
        # Multi-start optimization
        for _ in range(self.n_restarts_optimizer):
            # Random starting point
            x0 = np.random.uniform(0, 1, len(bounds))
            
            # Optimize acquisition function
            result = minimize(
                fun=lambda x: -self._acquisition_function(x.reshape(1, -1)),
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success and -result.fun > best_acquisition:
                best_acquisition = -result.fun
                best_x = result.x
        
        # Decode best point
        if best_x is not None:
            return self.space_manager.decode_parameters(best_x)
        else:
            # Fallback to random sampling
            logger.warning("Acquisition optimization failed, falling back to random sampling")
            return self.space_manager.sample_parameters(n_samples=1)[0]
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Compute acquisition function values"""
        if self.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            return self._expected_improvement(X)
        elif self.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
            return self._upper_confidence_bound(X)
        elif self.acquisition_function == AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT:
            return self._probability_of_improvement(X)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
    
    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function"""
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        if len(self.history.trials) == 0:
            return np.ones(X.shape[0])
        
        # Get current best
        f_best = max(trial.metrics.accuracy for trial in self.history.trials 
                    if trial.status == "completed" and trial.metrics)
        
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)
        
        # Compute EI
        z = (mu - f_best - self.xi) / sigma
        ei = (mu - f_best - self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        
        return ei
    
    def _upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition function"""
        mu, sigma = self.gp_model.predict(X, return_std=True)
        return mu + self.kappa * sigma
    
    def _probability_of_improvement(self, X: np.ndarray) -> np.ndarray:
        """Probability of Improvement acquisition function"""
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        if len(self.history.trials) == 0:
            return np.ones(X.shape[0])
        
        # Get current best
        f_best = max(trial.metrics.accuracy for trial in self.history.trials 
                    if trial.status == "completed" and trial.metrics)
        
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)
        
        # Compute PI
        z = (mu - f_best - self.xi) / sigma
        pi = stats.norm.cdf(z)
        
        return pi
    
    def get_optimization_history(self) -> OptimizationHistory:
        """Get optimization history"""
        return self.history
    
    def predict(self, parameters: Dict[str, Any]) -> Tuple[float, float]:
        """Predict performance for given parameters"""
        if self.gp_model is None:
            raise ValueError("Model not trained. Run optimization first.")
        
        encoded_params = self.space_manager.encode_parameters(parameters)
        mu, sigma = self.gp_model.predict(encoded_params.reshape(1, -1), return_std=True)
        
        return float(mu[0]), float(sigma[0])




class TreeStructuredParzenEstimator:
    """Tree-structured Parzen Estimator for hyperparameter optimization"""
    
    def __init__(
        self,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: float = 0.25,
        prior_weight: float = 1.0,
        consider_prior: bool = True,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False
    ):
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma  # Percentile for splitting good/bad trials
        self.prior_weight = prior_weight
        self.consider_prior = consider_prior
        self.consider_magic_clip = consider_magic_clip
        self.consider_endpoints = consider_endpoints
        
        self.history = OptimizationHistory()
        self.space_manager = None
        
        logger.info(f"Initialized TPE optimizer with gamma={gamma}")
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        space_manager: HyperparameterSpaceManager,
        max_trials: int = 50,
        convergence_patience: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using TPE
        
        Args:
            objective_function: Function to optimize (should return higher values for better performance)
            space_manager: Hyperparameter space manager
            max_trials: Maximum number of trials
            convergence_patience: Number of trials to wait for convergence
            
        Returns:
            Best hyperparameter configuration
        """
        self.space_manager = space_manager
        self.history = OptimizationHistory()
        
        logger.info(f"Starting TPE optimization with {max_trials} max trials")
        
        # Phase 1: Random exploration
        logger.info(f"Phase 1: Random exploration with {self.n_startup_trials} points")
        self._random_exploration(objective_function, self.n_startup_trials)
        
        # Phase 2: TPE optimization
        logger.info("Phase 2: TPE optimization")
        for trial_idx in range(self.n_startup_trials, max_trials):
            if self.history.has_converged(patience=convergence_patience):
                logger.info(f"Converged after {trial_idx} trials")
                break
            
            # Suggest next point using TPE
            next_params = self._suggest_next_point()
            
            # Evaluate objective function
            try:
                score = objective_function(next_params)
                
                # Create trial
                trial = Trial(
                    id=f"trial_{trial_idx}",
                    parameters=next_params,
                    metrics=PerformanceMetrics(
                        accuracy=score,
                        loss=1.0 - score,
                        precision=0.0,
                        recall=0.0,
                        f1_score=0.0,
                        training_time=0.0,
                        inference_time=0.0,
                        additional_metrics={}
                    ),
                    status="completed",
                    duration=0.0
                )
                
                self.history.add_trial(trial)
                logger.info(f"Trial {trial_idx}: score={score:.4f}, best={self.history.best_trial.metrics.accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {trial_idx} failed: {e}")
                # Add failed trial
                trial = Trial(
                    id=f"trial_{trial_idx}",
                    parameters=next_params,
                    metrics=None,
                    status="failed",
                    duration=0.0
                )
                self.history.add_trial(trial)
        
        best_params = self.history.get_best_parameters()
        logger.info(f"TPE optimization completed. Best score: {self.history.best_trial.metrics.accuracy:.4f}")
        
        return best_params
    
    def _random_exploration(self, objective_function: Callable, n_points: int) -> None:
        """Perform random exploration to initialize TPE"""
        if self.space_manager is None:
            raise ValueError("Space manager not set")
            
        for i in range(n_points):
            # Sample random parameters
            params = self.space_manager.sample_parameters(n_samples=1)[0]
            
            try:
                score = objective_function(params)
                
                trial = Trial(
                    id=f"random_{i}",
                    parameters=params,
                    metrics=PerformanceMetrics(
                        accuracy=score,
                        loss=1.0 - score,
                        precision=0.0,
                        recall=0.0,
                        f1_score=0.0,
                        training_time=0.0,
                        inference_time=0.0,
                        additional_metrics={}
                    ),
                    status="completed",
                    duration=0.0
                )
                
                self.history.add_trial(trial)
                logger.debug(f"Random exploration {i}: score={score:.4f}")
                
            except Exception as e:
                logger.error(f"Random exploration {i} failed: {e}")
    
    def _suggest_next_point(self) -> Dict[str, Any]:
        """Suggest next point using TPE algorithm"""
        # Get completed trials
        completed_trials = [t for t in self.history.trials if t.status == "completed" and t.metrics]
        
        if len(completed_trials) < 2:
            logger.warning("Not enough completed trials, falling back to random sampling")
            return self.space_manager.sample_parameters(n_samples=1)[0]
        
        # Sort trials by performance
        completed_trials.sort(key=lambda t: t.metrics.accuracy, reverse=True)
        
        # Split into good and bad trials
        n_good = max(1, int(self.gamma * len(completed_trials)))
        good_trials = completed_trials[:n_good]
        bad_trials = completed_trials[n_good:]
        
        logger.debug(f"Split trials: {len(good_trials)} good, {len(bad_trials)} bad")
        
        # Build density estimators for each parameter
        best_candidate = None
        best_ei = -np.inf
        
        # Generate candidate points
        for _ in range(self.n_ei_candidates):
            candidate = {}
            ei_value = 0.0
            
            # For each parameter, compute TPE expected improvement
            for param_name, param in self.space_manager.parameters.items():
                # Get parameter values from good and bad trials
                good_values = [t.parameters.get(param_name) for t in good_trials if param_name in t.parameters]
                bad_values = [t.parameters.get(param_name) for t in bad_trials if param_name in t.parameters]
                
                # Sample candidate value and compute EI
                candidate_value, param_ei = self._sample_parameter_with_ei(
                    param, good_values, bad_values
                )
                
                candidate[param_name] = candidate_value
                ei_value += param_ei
            
            # Keep track of best candidate
            if ei_value > best_ei:
                best_ei = ei_value
                best_candidate = candidate
        
        if best_candidate is None:
            logger.warning("TPE failed to generate candidate, falling back to random sampling")
            return self.space_manager.sample_parameters(n_samples=1)[0]
        
        return best_candidate
    
    def _sample_parameter_with_ei(
        self, 
        param: Parameter, 
        good_values: List[Any], 
        bad_values: List[Any]
    ) -> Tuple[Any, float]:
        """Sample parameter value and compute expected improvement"""
        
        if param.param_type == ParameterType.CONTINUOUS:
            return self._sample_continuous_with_ei(param, good_values, bad_values)
        elif param.param_type == ParameterType.INTEGER:
            return self._sample_integer_with_ei(param, good_values, bad_values)
        elif param.param_type in [ParameterType.CATEGORICAL, ParameterType.DISCRETE]:
            return self._sample_categorical_with_ei(param, good_values, bad_values)
        elif param.param_type == ParameterType.BOOLEAN:
            return self._sample_boolean_with_ei(param, good_values, bad_values)
        else:
            # Fallback to random sampling
            value = self.space_manager._sample_single_parameter(param)
            return value, 0.0
    
    def _sample_continuous_with_ei(
        self, 
        param: Parameter, 
        good_values: List[float], 
        bad_values: List[float]
    ) -> Tuple[float, float]:
        """Sample continuous parameter with expected improvement"""
        low, high = param.bounds
        
        if len(good_values) == 0:
            # No good values, sample uniformly
            if param.log_scale:
                value = np.exp(np.random.uniform(np.log(low), np.log(high)))
            else:
                value = np.random.uniform(low, high)
            return float(value), 0.0
        
        # Convert to log scale if needed
        if param.log_scale:
            good_values = [np.log(max(v, 1e-10)) for v in good_values if v is not None and v > 0]
            bad_values = [np.log(max(v, 1e-10)) for v in bad_values if v is not None and v > 0]
            low, high = np.log(low), np.log(high)
        else:
            good_values = [v for v in good_values if v is not None]
            bad_values = [v for v in bad_values if v is not None]
        
        if len(good_values) == 0:
            value = np.random.uniform(low, high)
            if param.log_scale:
                value = np.exp(value)
            return float(value), 0.0
        
        # Fit kernel density estimators
        good_kde = self._fit_kde(good_values, low, high)
        bad_kde = self._fit_kde(bad_values, low, high) if len(bad_values) > 0 else None
        
        # Sample from good distribution
        candidate = self._sample_from_kde(good_kde, low, high)
        
        # Compute expected improvement
        good_density = self._evaluate_kde(good_kde, candidate)
        bad_density = self._evaluate_kde(bad_kde, candidate) if bad_kde else 1e-10
        
        ei = good_density / max(bad_density, 1e-10)
        
        if param.log_scale:
            candidate = np.exp(candidate)
        
        return float(candidate), float(ei)
    
    def _sample_integer_with_ei(
        self, 
        param: Parameter, 
        good_values: List[int], 
        bad_values: List[int]
    ) -> Tuple[int, float]:
        """Sample integer parameter with expected improvement"""
        # Treat as continuous then round
        continuous_good = [float(v) for v in good_values if v is not None]
        continuous_bad = [float(v) for v in bad_values if v is not None]
        
        # Create temporary continuous parameter
        temp_param = Parameter(
            name=param.name,
            param_type=ParameterType.CONTINUOUS,
            bounds=(float(param.bounds[0]), float(param.bounds[1])),
            log_scale=param.log_scale
        )
        
        value, ei = self._sample_continuous_with_ei(temp_param, continuous_good, continuous_bad)
        
        # Round and clamp to bounds
        int_value = int(np.round(value))
        int_value = max(param.bounds[0], min(param.bounds[1], int_value))
        
        return int_value, ei
    
    def _sample_categorical_with_ei(
        self, 
        param: Parameter, 
        good_values: List[Any], 
        bad_values: List[Any]
    ) -> Tuple[Any, float]:
        """Sample categorical parameter with expected improvement"""
        categories = param.bounds
        
        if len(good_values) == 0:
            return random.choice(categories), 0.0
        
        # Count occurrences in good and bad trials
        good_counts = {cat: 0 for cat in categories}
        bad_counts = {cat: 0 for cat in categories}
        
        for val in good_values:
            if val in good_counts:
                good_counts[val] += 1
        
        for val in bad_values:
            if val in bad_counts:
                bad_counts[val] += 1
        
        # Add prior
        if self.consider_prior:
            for cat in categories:
                good_counts[cat] += self.prior_weight
                bad_counts[cat] += self.prior_weight
        
        # Compute probabilities
        total_good = sum(good_counts.values())
        total_bad = sum(bad_counts.values())
        
        # Compute expected improvement for each category
        ei_values = {}
        for cat in categories:
            good_prob = good_counts[cat] / max(total_good, 1)
            bad_prob = bad_counts[cat] / max(total_bad, 1)
            ei_values[cat] = good_prob / max(bad_prob, 1e-10)
        
        # Sample proportional to EI
        categories_list = list(categories)
        ei_list = [ei_values[cat] for cat in categories_list]
        
        # Normalize probabilities
        total_ei = sum(ei_list)
        if total_ei > 0:
            probs = [ei / total_ei for ei in ei_list]
        else:
            probs = [1.0 / len(categories_list)] * len(categories_list)
        
        # Sample category
        selected_idx = np.random.choice(len(categories_list), p=probs)
        selected_cat = categories_list[selected_idx]
        
        return selected_cat, ei_values[selected_cat]
    
    def _sample_boolean_with_ei(
        self, 
        param: Parameter, 
        good_values: List[bool], 
        bad_values: List[bool]
    ) -> Tuple[bool, float]:
        """Sample boolean parameter with expected improvement"""
        # Treat as categorical with [True, False]
        temp_param = Parameter(
            name=param.name,
            param_type=ParameterType.CATEGORICAL,
            bounds=[True, False]
        )
        
        return self._sample_categorical_with_ei(temp_param, good_values, bad_values)
    
    def _fit_kde(self, values: List[float], low: float, high: float) -> Dict[str, Any]:
        """Fit kernel density estimator"""
        if len(values) == 0:
            return {'type': 'uniform', 'low': low, 'high': high}
        
        if len(values) == 1:
            # Single point - use small gaussian around it
            return {
                'type': 'gaussian',
                'mean': values[0],
                'std': (high - low) * 0.1,
                'low': low,
                'high': high
            }
        
        # Multiple points - fit gaussian to data
        mean = np.mean(values)
        std = np.std(values)
        
        # Add some minimum bandwidth
        min_std = (high - low) * 0.01
        std = max(std, min_std)
        
        return {
            'type': 'gaussian',
            'mean': mean,
            'std': std,
            'low': low,
            'high': high
        }
    
    def _sample_from_kde(self, kde: Dict[str, Any], low: float, high: float) -> float:
        """Sample from kernel density estimator"""
        if kde['type'] == 'uniform':
            return np.random.uniform(low, high)
        elif kde['type'] == 'gaussian':
            # Sample from truncated gaussian
            for _ in range(100):  # Max attempts
                sample = np.random.normal(kde['mean'], kde['std'])
                if low <= sample <= high:
                    return sample
            # Fallback to uniform if truncated sampling fails
            return np.random.uniform(low, high)
        else:
            return np.random.uniform(low, high)
    
    def _evaluate_kde(self, kde: Optional[Dict[str, Any]], x: float) -> float:
        """Evaluate kernel density estimator at point x"""
        if kde is None:
            return 1e-10
        
        if kde['type'] == 'uniform':
            low, high = kde['low'], kde['high']
            return 1.0 / (high - low) if low <= x <= high else 1e-10
        elif kde['type'] == 'gaussian':
            mean, std = kde['mean'], kde['std']
            # Gaussian PDF
            return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        else:
            return 1e-10
    
    def get_optimization_history(self) -> OptimizationHistory:
        """Get optimization history"""
        return self.history


# Update the main service to support TPE
class HyperparameterOptimizationService(IHyperparameterOptimizer):
    """Main hyperparameter optimization service with multiple algorithms"""
    
    def __init__(self, algorithm: str = "bayesian"):
        self.algorithm = algorithm
        self.current_space = None
        self.optimizer = None
        self.history = OptimizationHistory()
        
        logger.info(f"Initialized HyperparameterOptimizationService with {algorithm} algorithm")
    
    def define_search_space(self, architecture: Architecture) -> HyperparameterSpace:
        """Define hyperparameter search space for architecture"""
        space_manager = create_default_hyperparameter_space(architecture)
        self.current_space = space_manager
        
        # Convert to interface format
        hyperparameter_space = HyperparameterSpace(
            parameters={
                name: (param.bounds[0] if param.param_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER] else param.bounds,
                       param.bounds[1] if param.param_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER] else param.bounds)
                for name, param in space_manager.parameters.items()
            },
            parameter_types={
                name: param.param_type.value
                for name, param in space_manager.parameters.items()
            },
            constraints=[
                {
                    'type': constraint.constraint_type,
                    'parameters': constraint.parameters,
                    'condition': constraint.condition,
                    'action': constraint.action
                }
                for constraint in space_manager.constraints
            ]
        )
        
        logger.info(f"Defined search space with {len(space_manager.parameters)} parameters")
        return hyperparameter_space
    
    def optimize(
        self,
        objective_function: Callable,
        search_space: HyperparameterSpace,
        max_trials: int = 50
    ) -> TrainingConfig:
        """Optimize hyperparameters using selected algorithm"""
        if self.current_space is None:
            raise ValueError("Search space not defined. Call define_search_space first.")
        
        # Initialize optimizer based on algorithm
        if self.algorithm == "bayesian":
            self.optimizer = BayesianOptimizer(
                acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
                n_initial_points=min(5, max_trials // 10),
                xi=0.01,
                kappa=2.576
            )
        elif self.algorithm == "tpe":
            self.optimizer = TreeStructuredParzenEstimator(
                n_startup_trials=min(10, max_trials // 5),
                n_ei_candidates=24,
                gamma=0.25
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Run optimization
        best_params = self.optimizer.optimize(
            objective_function=objective_function,
            space_manager=self.current_space,
            max_trials=max_trials
        )
        
        # Convert to TrainingConfig
        training_config = TrainingConfig(
            batch_size=best_params.get('batch_size', 32),
            learning_rate=best_params.get('learning_rate', 1e-3),
            optimizer=best_params.get('optimizer', 'adam'),
            epochs=best_params.get('epochs', 100),
            early_stopping_patience=best_params.get('early_stopping_patience', 10),
            regularization={
                'weight_decay': best_params.get('weight_decay', 1e-4),
                'dropout_rate': best_params.get('dropout_rate', 0.1)
            },
            use_mixed_precision=best_params.get('use_mixed_precision', False)
        )
        
        # Update history
        self.history = self.optimizer.get_optimization_history()
        
        logger.info(f"Optimization completed. Best configuration: {training_config}")
        return training_config
    
    def get_optimization_history(self) -> List[Trial]:
        """Get history of optimization trials"""
        return self.history.trials
    
    def suggest_next_parameters(self) -> Dict[str, Any]:
        """Suggest next parameters to evaluate"""
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized. Run optimize first.")
        
        if hasattr(self.optimizer, '_suggest_next_point'):
            return self.optimizer._suggest_next_point()
        else:
            raise ValueError("Current optimizer does not support parameter suggestion")
    
    def update_trial_result(self, parameters: Dict[str, Any], metrics: PerformanceMetrics) -> None:
        """Update trial result for online optimization"""
        trial = Trial(
            id=f"trial_{len(self.history.trials)}",
            parameters=parameters,
            metrics=metrics,
            status="completed",
            duration=0.0
        )
        
        self.history.add_trial(trial)
        logger.info(f"Updated trial result: accuracy={metrics.accuracy:.4f}")