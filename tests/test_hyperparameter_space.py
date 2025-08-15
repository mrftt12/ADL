"""
Tests for hyperparameter space definition and management
"""

import pytest
import numpy as np
from unittest.mock import Mock

from automl_framework.services.hyperparameter_optimization import (
    HyperparameterSpaceManager,
    Parameter,
    ParameterType,
    ParameterConstraint,
    create_default_hyperparameter_space
)
from automl_framework.core.interfaces import Architecture, Layer


class TestParameter:
    """Test Parameter class"""
    
    def test_parameter_creation(self):
        """Test parameter creation with different types"""
        # Continuous parameter
        param = Parameter(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-1),
            log_scale=True,
            default=1e-3
        )
        
        assert param.name == "learning_rate"
        assert param.param_type == ParameterType.CONTINUOUS
        assert param.bounds == (1e-5, 1e-1)
        assert param.log_scale is True
        assert param.default == 1e-3
    
    def test_categorical_parameter(self):
        """Test categorical parameter creation"""
        param = Parameter(
            name="optimizer",
            param_type=ParameterType.CATEGORICAL,
            bounds=["adam", "sgd", "rmsprop"],
            default="adam"
        )
        
        assert param.param_type == ParameterType.CATEGORICAL
        assert param.bounds == ["adam", "sgd", "rmsprop"]
        assert param.default == "adam"


class TestHyperparameterSpaceManager:
    """Test HyperparameterSpaceManager class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.space = HyperparameterSpaceManager()
    
    def test_add_parameter(self):
        """Test adding parameters to space"""
        param = Parameter(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-1),
            log_scale=True
        )
        
        self.space.add_parameter(param)
        
        assert "learning_rate" in self.space.parameters
        assert self.space.parameters["learning_rate"] == param
    
    def test_add_constraint(self):
        """Test adding constraints to space"""
        constraint = ParameterConstraint(
            constraint_type="conditional",
            parameters=["optimizer", "momentum"],
            condition="optimizer == sgd",
            action="momentum = 0.9"
        )
        
        self.space.add_constraint(constraint)
        
        assert len(self.space.constraints) == 1
        assert self.space.constraints[0] == constraint
    
    def test_sample_continuous_parameter(self):
        """Test sampling continuous parameters"""
        param = Parameter(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-1),
            log_scale=True
        )
        self.space.add_parameter(param)
        
        samples = self.space.sample_parameters(n_samples=100)
        
        assert len(samples) == 100
        for sample in samples:
            assert "learning_rate" in sample
            assert 1e-5 <= sample["learning_rate"] <= 1e-1
    
    def test_sample_categorical_parameter(self):
        """Test sampling categorical parameters"""
        param = Parameter(
            name="optimizer",
            param_type=ParameterType.CATEGORICAL,
            bounds=["adam", "sgd", "rmsprop"]
        )
        self.space.add_parameter(param)
        
        samples = self.space.sample_parameters(n_samples=100)
        
        assert len(samples) == 100
        for sample in samples:
            assert "optimizer" in sample
            assert sample["optimizer"] in ["adam", "sgd", "rmsprop"]
    
    def test_sample_integer_parameter(self):
        """Test sampling integer parameters"""
        param = Parameter(
            name="batch_size",
            param_type=ParameterType.INTEGER,
            bounds=(16, 256)
        )
        self.space.add_parameter(param)
        
        samples = self.space.sample_parameters(n_samples=100)
        
        assert len(samples) == 100
        for sample in samples:
            assert "batch_size" in sample
            assert isinstance(sample["batch_size"], int)
            assert 16 <= sample["batch_size"] <= 256
    
    def test_sample_boolean_parameter(self):
        """Test sampling boolean parameters"""
        param = Parameter(
            name="use_mixed_precision",
            param_type=ParameterType.BOOLEAN,
            bounds=[True, False]
        )
        self.space.add_parameter(param)
        
        samples = self.space.sample_parameters(n_samples=100)
        
        assert len(samples) == 100
        for sample in samples:
            assert "use_mixed_precision" in sample
            assert isinstance(sample["use_mixed_precision"], bool)
    
    def test_encode_decode_parameters(self):
        """Test parameter encoding and decoding"""
        # Add various parameter types
        self.space.add_parameter(Parameter(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-1),
            log_scale=True
        ))
        
        self.space.add_parameter(Parameter(
            name="batch_size",
            param_type=ParameterType.INTEGER,
            bounds=(16, 256)
        ))
        
        self.space.add_parameter(Parameter(
            name="optimizer",
            param_type=ParameterType.CATEGORICAL,
            bounds=["adam", "sgd", "rmsprop"]
        ))
        
        self.space.add_parameter(Parameter(
            name="use_mixed_precision",
            param_type=ParameterType.BOOLEAN,
            bounds=[True, False]
        ))
        
        # Test encoding and decoding
        original_params = {
            "learning_rate": 1e-3,
            "batch_size": 64,
            "optimizer": "adam",
            "use_mixed_precision": True
        }
        
        encoded = self.space.encode_parameters(original_params)
        decoded = self.space.decode_parameters(encoded)
        
        # Check that decoding is approximately correct
        assert abs(decoded["learning_rate"] - original_params["learning_rate"]) < 1e-4
        assert decoded["batch_size"] == original_params["batch_size"]
        assert decoded["optimizer"] == original_params["optimizer"]
        assert decoded["use_mixed_precision"] == original_params["use_mixed_precision"]
    
    def test_get_bounds(self):
        """Test getting optimization bounds"""
        self.space.add_parameter(Parameter(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-1)
        ))
        
        self.space.add_parameter(Parameter(
            name="optimizer",
            param_type=ParameterType.CATEGORICAL,
            bounds=["adam", "sgd", "rmsprop"]
        ))
        
        bounds = self.space.get_bounds()
        
        # Should have 4 bounds: 1 for learning_rate + 3 for optimizer categories
        assert len(bounds) == 4
        for bound in bounds:
            assert bound == (0.0, 1.0)
    
    def test_validate_parameters(self):
        """Test parameter validation"""
        self.space.add_parameter(Parameter(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-1)
        ))
        
        self.space.add_parameter(Parameter(
            name="optimizer",
            param_type=ParameterType.CATEGORICAL,
            bounds=["adam", "sgd", "rmsprop"]
        ))
        
        # Valid parameters
        valid_params = {
            "learning_rate": 1e-3,
            "optimizer": "adam"
        }
        assert self.space.validate_parameters(valid_params) is True
        
        # Invalid learning rate (out of bounds)
        invalid_params = {
            "learning_rate": 1.0,  # Too high
            "optimizer": "adam"
        }
        assert self.space.validate_parameters(invalid_params) is False
        
        # Invalid optimizer
        invalid_params = {
            "learning_rate": 1e-3,
            "optimizer": "invalid_optimizer"
        }
        assert self.space.validate_parameters(invalid_params) is False
    
    def test_to_dict_from_dict(self):
        """Test serialization and deserialization"""
        # Create space with parameters and constraints
        self.space.add_parameter(Parameter(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-1),
            log_scale=True,
            default=1e-3
        ))
        
        self.space.add_parameter(Parameter(
            name="optimizer",
            param_type=ParameterType.CATEGORICAL,
            bounds=["adam", "sgd", "rmsprop"],
            default="adam"
        ))
        
        constraint = ParameterConstraint(
            constraint_type="conditional",
            parameters=["optimizer", "momentum"],
            condition="optimizer == sgd",
            action="momentum = 0.9"
        )
        self.space.add_constraint(constraint)
        
        # Convert to dict and back
        space_dict = self.space.to_dict()
        new_space = HyperparameterSpaceManager.from_dict(space_dict)
        
        # Check parameters
        assert len(new_space.parameters) == 2
        assert "learning_rate" in new_space.parameters
        assert "optimizer" in new_space.parameters
        
        lr_param = new_space.parameters["learning_rate"]
        assert lr_param.param_type == ParameterType.CONTINUOUS
        assert lr_param.bounds == (1e-5, 1e-1)
        assert lr_param.log_scale is True
        assert lr_param.default == 1e-3
        
        # Check constraints
        assert len(new_space.constraints) == 1
        assert new_space.constraints[0].constraint_type == "conditional"


class TestDefaultHyperparameterSpace:
    """Test default hyperparameter space creation"""
    
    def test_create_default_space(self):
        """Test creating default hyperparameter space"""
        # Create mock architecture
        architecture = Architecture(
            id="test_arch",
            layers=[],
            connections=[],
            input_shape=(224, 224, 3),
            output_shape=(10,),
            parameter_count=1000000,
            flops=1000000
        )
        
        space = create_default_hyperparameter_space(architecture)
        
        # Check that all expected parameters are present
        expected_params = [
            "learning_rate", "batch_size", "optimizer", "weight_decay",
            "dropout_rate", "early_stopping_patience", "epochs", "use_mixed_precision"
        ]
        
        for param_name in expected_params:
            assert param_name in space.parameters
        
        # Check parameter types
        assert space.parameters["learning_rate"].param_type == ParameterType.CONTINUOUS
        assert space.parameters["batch_size"].param_type == ParameterType.DISCRETE
        assert space.parameters["optimizer"].param_type == ParameterType.CATEGORICAL
        assert space.parameters["use_mixed_precision"].param_type == ParameterType.BOOLEAN
        
        # Check that constraints are added
        assert len(space.constraints) > 0
    
    def test_sample_from_default_space(self):
        """Test sampling from default hyperparameter space"""
        architecture = Architecture(
            id="test_arch",
            layers=[],
            connections=[],
            input_shape=(224, 224, 3),
            output_shape=(10,),
            parameter_count=1000000,
            flops=1000000
        )
        
        space = create_default_hyperparameter_space(architecture)
        samples = space.sample_parameters(n_samples=10)
        
        assert len(samples) == 10
        
        for sample in samples:
            # Check that all parameters are present
            assert "learning_rate" in sample
            assert "batch_size" in sample
            assert "optimizer" in sample
            assert "weight_decay" in sample
            assert "dropout_rate" in sample
            assert "early_stopping_patience" in sample
            assert "epochs" in sample
            assert "use_mixed_precision" in sample
            
            # Check value ranges
            assert 1e-5 <= sample["learning_rate"] <= 1e-1
            assert sample["batch_size"] in [16, 32, 64, 128, 256]
            assert sample["optimizer"] in ["adam", "sgd", "rmsprop", "adamw"]
            assert 1e-6 <= sample["weight_decay"] <= 1e-2
            assert 0.0 <= sample["dropout_rate"] <= 0.8
            assert 5 <= sample["early_stopping_patience"] <= 50
            assert 10 <= sample["epochs"] <= 200
            assert isinstance(sample["use_mixed_precision"], bool)


if __name__ == "__main__":
    pytest.main([__file__])