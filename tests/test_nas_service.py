"""
Unit tests for Neural Architecture Search (NAS) service.

Tests cover architecture representation, search space definition,
encoding/decoding, validation, and parameter counting.
"""

import pytest
import uuid
from copy import deepcopy
from unittest.mock import Mock, patch

from automl_framework.services.nas_service import (
    NASService, SearchSpace, LayerSpec, ArchitectureEncoder, 
    ArchitectureValidator, ParameterCounter, ActivationFunction, PoolingType
)
from automl_framework.models.data_models import (
    Architecture, Layer, Connection, LayerType, TaskType, DataType
)
from automl_framework.models.data_models import PerformanceMetrics


class TestLayerSpec:
    """Test LayerSpec class."""
    
    def test_layer_spec_creation(self):
        """Test creating a valid layer specification."""
        spec = LayerSpec(
            layer_type=LayerType.DENSE,
            parameter_ranges={
                'units': (32, 512),
                'activation': ['relu', 'tanh']
            }
        )
        
        assert spec.layer_type == LayerType.DENSE
        assert spec.parameter_ranges['units'] == (32, 512)
        assert spec.parameter_ranges['activation'] == ['relu', 'tanh']
    
    def test_layer_spec_validation_success(self):
        """Test successful layer spec validation."""
        spec = LayerSpec(
            layer_type=LayerType.CONV2D,
            parameter_ranges={
                'filters': (16, 256),
                'kernel_size': (1, 7)
            }
        )
        
        # Should not raise any exception
        spec.validate()
    
    def test_layer_spec_validation_invalid_layer_type(self):
        """Test layer spec validation with invalid layer type."""
        with pytest.raises(ValueError, match="layer_type must be a LayerType enum"):
            spec = LayerSpec(
                layer_type="invalid",
                parameter_ranges={}
            )
            spec.validate()
    
    def test_layer_spec_validation_invalid_parameter_ranges(self):
        """Test layer spec validation with invalid parameter ranges."""
        with pytest.raises(ValueError, match="parameter_ranges must be a dictionary"):
            spec = LayerSpec(
                layer_type=LayerType.DENSE,
                parameter_ranges="invalid"
            )
            spec.validate()
    
    def test_layer_spec_validation_invalid_range_values(self):
        """Test layer spec validation with invalid range values."""
        with pytest.raises(ValueError, match="Invalid range"):
            spec = LayerSpec(
                layer_type=LayerType.DENSE,
                parameter_ranges={
                    'units': (512, 32)  # min > max
                }
            )
            spec.validate()


class TestSearchSpace:
    """Test SearchSpace class."""
    
    def test_search_space_creation(self):
        """Test creating a valid search space."""
        layer_specs = [
            LayerSpec(LayerType.DENSE, {'units': (32, 512)}),
            LayerSpec(LayerType.DROPOUT, {'rate': (0.1, 0.5)})
        ]
        
        search_space = SearchSpace(
            layer_specs=layer_specs,
            max_layers=10,
            min_layers=2
        )
        
        assert len(search_space.layer_specs) == 2
        assert search_space.max_layers == 10
        assert search_space.min_layers == 2
    
    def test_search_space_validation_success(self):
        """Test successful search space validation."""
        layer_specs = [
            LayerSpec(LayerType.DENSE, {'units': (32, 512)})
        ]
        
        search_space = SearchSpace(
            layer_specs=layer_specs,
            max_layers=5,
            min_layers=1
        )
        
        # Should not raise any exception
        search_space.validate()
    
    def test_search_space_validation_empty_layer_specs(self):
        """Test search space validation with empty layer specs."""
        with pytest.raises(ValueError, match="Search space must have at least one layer specification"):
            search_space = SearchSpace(
                layer_specs=[],
                max_layers=5
            )
            search_space.validate()
    
    def test_search_space_validation_invalid_layer_counts(self):
        """Test search space validation with invalid layer counts."""
        layer_specs = [LayerSpec(LayerType.DENSE, {'units': (32, 512)})]
        
        with pytest.raises(ValueError, match="max_layers must be positive"):
            SearchSpace(layer_specs=layer_specs, max_layers=0).validate()
        
        with pytest.raises(ValueError, match="min_layers must be positive"):
            SearchSpace(layer_specs=layer_specs, max_layers=5, min_layers=0).validate()
        
        with pytest.raises(ValueError, match="min_layers cannot be greater than max_layers"):
            SearchSpace(layer_specs=layer_specs, max_layers=3, min_layers=5).validate()
    
    def test_get_layer_types(self):
        """Test getting layer types from search space."""
        layer_specs = [
            LayerSpec(LayerType.DENSE, {'units': (32, 512)}),
            LayerSpec(LayerType.CONV2D, {'filters': (16, 256)}),
            LayerSpec(LayerType.DROPOUT, {'rate': (0.1, 0.5)})
        ]
        
        search_space = SearchSpace(layer_specs=layer_specs, max_layers=10)
        layer_types = search_space.get_layer_types()
        
        assert LayerType.DENSE in layer_types
        assert LayerType.CONV2D in layer_types
        assert LayerType.DROPOUT in layer_types
        assert len(layer_types) == 3
    
    def test_get_layer_spec(self):
        """Test getting specific layer spec from search space."""
        dense_spec = LayerSpec(LayerType.DENSE, {'units': (32, 512)})
        conv_spec = LayerSpec(LayerType.CONV2D, {'filters': (16, 256)})
        
        search_space = SearchSpace(
            layer_specs=[dense_spec, conv_spec],
            max_layers=10
        )
        
        retrieved_dense = search_space.get_layer_spec(LayerType.DENSE)
        retrieved_conv = search_space.get_layer_spec(LayerType.CONV2D)
        retrieved_none = search_space.get_layer_spec(LayerType.LSTM)
        
        assert retrieved_dense == dense_spec
        assert retrieved_conv == conv_spec
        assert retrieved_none is None


class TestArchitectureEncoder:
    """Test ArchitectureEncoder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.layer_specs = [
            LayerSpec(LayerType.DENSE, {'units': (32, 512)}),
            LayerSpec(LayerType.DROPOUT, {'rate': (0.1, 0.5)})
        ]
        self.search_space = SearchSpace(
            layer_specs=self.layer_specs,
            max_layers=5,
            input_shape=(784,),
            output_shape=(10,)
        )
        self.encoder = ArchitectureEncoder(self.search_space)
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        assert self.encoder.search_space == self.search_space
        assert len(self.encoder.layer_type_to_id) == len(LayerType)
        assert len(self.encoder.id_to_layer_type) == len(LayerType)
    
    def test_encode_decode_architecture(self):
        """Test encoding and decoding architecture."""
        # Create test architecture
        layers = [
            Layer(LayerType.DENSE, {'units': 128, 'activation': 'relu'}),
            Layer(LayerType.DROPOUT, {'rate': 0.3}),
            Layer(LayerType.DENSE, {'units': 10, 'activation': 'softmax'})
        ]
        connections = [
            Connection(0, 1),
            Connection(1, 2)
        ]
        
        original_arch = Architecture(
            id="test-arch",
            layers=layers,
            connections=connections,
            input_shape=(784,),
            output_shape=(10,),
            parameter_count=1000,
            flops=5000,
            task_type=TaskType.CLASSIFICATION
        )
        
        # Encode and decode
        encoded = self.encoder.encode_architecture(original_arch)
        decoded = self.encoder.decode_architecture(encoded)
        
        # Verify decoded architecture matches original
        assert decoded.id == original_arch.id
        assert len(decoded.layers) == len(original_arch.layers)
        assert len(decoded.connections) == len(original_arch.connections)
        assert decoded.input_shape == original_arch.input_shape
        assert decoded.output_shape == original_arch.output_shape
        assert decoded.parameter_count == original_arch.parameter_count
        assert decoded.flops == original_arch.flops
        assert decoded.task_type == original_arch.task_type
    
    def test_encode_to_vector(self):
        """Test encoding architecture to numerical vector."""
        layers = [
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DROPOUT, {'rate': 0.3})
        ]
        
        architecture = Architecture(
            id="test-arch",
            layers=layers,
            connections=[Connection(0, 1)],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        vector = self.encoder.encode_to_vector(architecture)
        
        # Vector should be a list of floats
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
        assert len(vector) > 0
        
        # First element should be normalized layer count
        expected_layer_ratio = len(layers) / self.search_space.max_layers
        assert vector[0] == expected_layer_ratio
    
    def test_generate_random_architecture(self):
        """Test generating random valid architecture."""
        architecture = self.encoder._generate_random_architecture()
        
        # Verify basic properties
        assert isinstance(architecture, Architecture)
        assert len(architecture.layers) >= self.search_space.min_layers
        assert len(architecture.layers) <= self.search_space.max_layers
        assert architecture.input_shape == self.search_space.input_shape
        assert architecture.output_shape == self.search_space.output_shape
        
        # Verify all layers are valid
        available_types = self.search_space.get_layer_types()
        for layer in architecture.layers:
            assert layer.layer_type in available_types
        
        # Verify connections exist for multi-layer architectures
        if len(architecture.layers) > 1:
            assert len(architecture.connections) >= len(architecture.layers) - 1


class TestArchitectureValidator:
    """Test ArchitectureValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.layer_specs = [
            LayerSpec(LayerType.DENSE, {'units': (32, 512)}),
            LayerSpec(LayerType.DROPOUT, {'rate': (0.1, 0.5)})
        ]
        self.search_space = SearchSpace(
            layer_specs=self.layer_specs,
            max_layers=5,
            min_layers=2
        )
        self.validator = ArchitectureValidator(self.search_space)
    
    def test_validate_valid_architecture(self):
        """Test validating a valid architecture."""
        layers = [
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DROPOUT, {'rate': 0.3}),
            Layer(LayerType.DENSE, {'units': 64})
        ]
        connections = [Connection(0, 1), Connection(1, 2)]
        
        architecture = Architecture(
            id="valid-arch",
            layers=layers,
            connections=connections,
            input_shape=(784,),
            output_shape=(10,)
        )
        
        is_valid, errors = self.validator.validate_architecture(architecture)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_too_few_layers(self):
        """Test validating architecture with too few layers."""
        layers = [Layer(LayerType.DENSE, {'units': 128})]
        
        architecture = Architecture(
            id="few-layers-arch",
            layers=layers,
            connections=[],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        is_valid, errors = self.validator.validate_architecture(architecture)
        
        assert not is_valid
        assert any("Too few layers" in error for error in errors)
    
    def test_validate_too_many_layers(self):
        """Test validating architecture with too many layers."""
        layers = [Layer(LayerType.DENSE, {'units': 128}) for _ in range(10)]
        connections = [Connection(i, i+1) for i in range(9)]
        
        architecture = Architecture(
            id="many-layers-arch",
            layers=layers,
            connections=connections,
            input_shape=(784,),
            output_shape=(10,)
        )
        
        is_valid, errors = self.validator.validate_architecture(architecture)
        
        assert not is_valid
        assert any("Too many layers" in error for error in errors)
    
    def test_validate_invalid_layer_type(self):
        """Test validating architecture with invalid layer type."""
        layers = [
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.LSTM, {'units': 64})  # LSTM not in search space
        ]
        connections = [Connection(0, 1)]
        
        architecture = Architecture(
            id="invalid-type-arch",
            layers=layers,
            connections=connections,
            input_shape=(784,),
            output_shape=(10,)
        )
        
        is_valid, errors = self.validator.validate_architecture(architecture)
        
        assert not is_valid
        assert any("not in search space" in error for error in errors)
    
    def test_validate_parameter_out_of_range(self):
        """Test validating architecture with parameters out of range."""
        layers = [
            Layer(LayerType.DENSE, {'units': 1000}),  # Out of range (32, 512)
            Layer(LayerType.DROPOUT, {'rate': 0.3})
        ]
        connections = [Connection(0, 1)]
        
        architecture = Architecture(
            id="out-of-range-arch",
            layers=layers,
            connections=connections,
            input_shape=(784,),
            output_shape=(10,)
        )
        
        is_valid, errors = self.validator.validate_architecture(architecture)
        
        assert not is_valid
        assert any("out of range" in error for error in errors)
    
    def test_validate_missing_connections(self):
        """Test validating multi-layer architecture without connections."""
        layers = [
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DENSE, {'units': 64})
        ]
        
        architecture = Architecture(
            id="no-connections-arch",
            layers=layers,
            connections=[],  # No connections
            input_shape=(784,),
            output_shape=(10,)
        )
        
        is_valid, errors = self.validator.validate_architecture(architecture)
        
        assert not is_valid
        assert any("must have connections" in error for error in errors)


class TestParameterCounter:
    """Test ParameterCounter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.counter = ParameterCounter()
    
    def test_count_dense_layer_parameters(self):
        """Test counting parameters for dense layer."""
        layer = Layer(
            LayerType.DENSE,
            {
                'input_size': 784,
                'units': 128,
                'use_bias': True
            }
        )
        
        params = self.counter._count_layer_parameters(layer)
        expected = 784 * 128 + 128  # weights + bias
        
        assert params == expected
    
    def test_count_dense_layer_parameters_no_bias(self):
        """Test counting parameters for dense layer without bias."""
        layer = Layer(
            LayerType.DENSE,
            {
                'input_size': 784,
                'units': 128,
                'use_bias': False
            }
        )
        
        params = self.counter._count_layer_parameters(layer)
        expected = 784 * 128  # weights only
        
        assert params == expected
    
    def test_count_conv2d_layer_parameters(self):
        """Test counting parameters for conv2d layer."""
        layer = Layer(
            LayerType.CONV2D,
            {
                'kernel_size': (3, 3),
                'input_channels': 32,
                'filters': 64,
                'use_bias': True
            }
        )
        
        params = self.counter._count_layer_parameters(layer)
        expected = 3 * 3 * 32 * 64 + 64  # kernel weights + bias
        
        assert params == expected
    
    def test_count_lstm_layer_parameters(self):
        """Test counting parameters for LSTM layer."""
        layer = Layer(
            LayerType.LSTM,
            {
                'input_size': 100,
                'units': 128
            }
        )
        
        params = self.counter._count_layer_parameters(layer)
        # LSTM has 4 gates: input, forget, cell, output
        # Each gate has input weights, hidden weights, and bias
        expected = 4 * (100 * 128 + 128 * 128 + 128)
        
        assert params == expected
    
    def test_count_batch_norm_parameters(self):
        """Test counting parameters for batch normalization layer."""
        layer = Layer(
            LayerType.BATCH_NORM,
            {'num_features': 64}
        )
        
        params = self.counter._count_layer_parameters(layer)
        expected = 2 * 64  # gamma and beta
        
        assert params == expected
    
    def test_count_architecture_parameters(self):
        """Test counting total parameters for architecture."""
        layers = [
            Layer(LayerType.DENSE, {'input_size': 784, 'units': 128, 'use_bias': True}),
            Layer(LayerType.DROPOUT, {'rate': 0.3}),  # No parameters
            Layer(LayerType.DENSE, {'input_size': 128, 'units': 10, 'use_bias': True})
        ]
        
        architecture = Architecture(
            id="test-arch",
            layers=layers,
            connections=[Connection(0, 1), Connection(1, 2)],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        total_params = self.counter.count_parameters(architecture)
        expected = (784 * 128 + 128) + 0 + (128 * 10 + 10)
        
        assert total_params == expected
    
    def test_count_dense_layer_flops(self):
        """Test counting FLOPs for dense layer."""
        layer = Layer(
            LayerType.DENSE,
            {
                'input_size': 784,
                'units': 128,
                'batch_size': 32
            }
        )
        
        flops = self.counter._count_layer_flops(layer)
        expected = 32 * 784 * 128
        
        assert flops == expected
    
    def test_count_conv2d_layer_flops(self):
        """Test counting FLOPs for conv2d layer."""
        layer = Layer(
            LayerType.CONV2D,
            {
                'kernel_size': (3, 3),
                'input_channels': 32,
                'filters': 64,
                'output_height': 28,
                'output_width': 28,
                'batch_size': 32
            }
        )
        
        flops = self.counter._count_layer_flops(layer)
        expected = 32 * 28 * 28 * 3 * 3 * 32 * 64
        
        assert flops == expected


class TestDARTSSearcher:
    """Test DARTSSearcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from automl_framework.services.nas_service import DARTSSearcher
        
        self.layer_specs = [
            LayerSpec(LayerType.DENSE, {'units': (32, 512)}),
            LayerSpec(LayerType.CONV2D, {'filters': (16, 256)}),
            LayerSpec(LayerType.DROPOUT, {'rate': (0.1, 0.5)})
        ]
        self.search_space = SearchSpace(
            layer_specs=self.layer_specs,
            max_layers=10,
            min_layers=2,
            input_shape=(32, 32, 3),
            output_shape=(10,),
            task_type=TaskType.CLASSIFICATION
        )
        self.darts_searcher = DARTSSearcher(self.search_space)
    
    def test_darts_searcher_initialization(self):
        """Test DARTS searcher initialization."""
        assert self.darts_searcher.search_space == self.search_space
        assert self.darts_searcher.num_nodes == 4
        assert self.darts_searcher.num_ops > 0
        assert len(self.darts_searcher.alpha_normal) > 0
        assert len(self.darts_searcher.alpha_reduce) > 0
    
    def test_get_primitive_operations(self):
        """Test getting primitive operations."""
        operations = self.darts_searcher._get_primitive_operations()
        
        assert isinstance(operations, list)
        assert len(operations) > 0
        assert 'none' in operations
        assert 'skip_connect' in operations
        assert 'sep_conv_3x3' in operations
        assert 'max_pool_3x3' in operations
    
    def test_softmax_function(self):
        """Test softmax function."""
        import numpy as np
        
        logits = np.array([1.0, 2.0, 3.0])
        probs = self.darts_searcher._softmax(logits)
        
        assert len(probs) == len(logits)
        assert abs(np.sum(probs) - 1.0) < 1e-6  # Sum should be 1
        assert all(p >= 0 for p in probs)  # All probabilities should be non-negative
        assert probs[2] > probs[1] > probs[0]  # Higher logits should have higher probs
    
    def test_gumbel_softmax(self):
        """Test Gumbel softmax function."""
        import numpy as np
        
        logits = np.array([1.0, 2.0, 3.0])
        probs = self.darts_searcher._gumbel_softmax(logits, temperature=1.0)
        
        assert len(probs) == len(logits)
        assert abs(np.sum(probs) - 1.0) < 1e-6  # Sum should be 1
        assert all(p >= 0 for p in probs)  # All probabilities should be non-negative
    
    def test_get_continuous_architecture(self):
        """Test getting continuous architecture representation."""
        continuous_arch = self.darts_searcher.get_continuous_architecture('normal')
        
        assert isinstance(continuous_arch, dict)
        assert len(continuous_arch) == self.darts_searcher.num_nodes
        
        for node_name, node_ops in continuous_arch.items():
            assert node_name.startswith('node_')
            assert isinstance(node_ops, dict)
            
            for input_name, weighted_ops in node_ops.items():
                assert input_name.startswith('input_')
                assert isinstance(weighted_ops, dict)
                
                # Check that probabilities sum to approximately 1
                total_prob = sum(weighted_ops.values())
                assert abs(total_prob - 1.0) < 1e-5
    
    def test_discretize_architecture(self):
        """Test discretizing continuous architecture."""
        architecture = self.darts_searcher.discretize_architecture(temperature=0.1)
        
        assert isinstance(architecture, Architecture)
        assert len(architecture.layers) > 0
        assert architecture.input_shape == self.search_space.input_shape
        assert architecture.output_shape == self.search_space.output_shape
        assert architecture.task_type == self.search_space.task_type
    
    def test_operation_to_layer(self):
        """Test converting operations to layers."""
        # Test convolution operation
        conv_layer = self.darts_searcher._operation_to_layer('sep_conv_3x3')
        assert conv_layer is not None
        assert conv_layer.layer_type == LayerType.CONV2D
        assert conv_layer.parameters['kernel_size'] == (3, 3)
        
        # Test pooling operation
        pool_layer = self.darts_searcher._operation_to_layer('max_pool_3x3')
        assert pool_layer is not None
        assert pool_layer.layer_type == LayerType.POOLING
        assert pool_layer.parameters['pool_type'] == 'max'
        
        # Test dense operation
        dense_layer = self.darts_searcher._operation_to_layer('dense_128')
        assert dense_layer is not None
        assert dense_layer.layer_type == LayerType.DENSE
        assert dense_layer.parameters['units'] == 128
        
        # Test skip connection (should return None)
        skip_layer = self.darts_searcher._operation_to_layer('skip_connect')
        assert skip_layer is None
    
    def test_search_architectures(self):
        """Test DARTS architecture search."""
        architectures = self.darts_searcher.search_architectures(
            num_epochs=10,  # Reduced for testing
            num_architectures=3
        )
        
        assert isinstance(architectures, list)
        assert len(architectures) <= 3
        
        for arch in architectures:
            assert isinstance(arch, Architecture)
            assert len(arch.layers) > 0
            assert arch.input_shape == self.search_space.input_shape
            assert arch.output_shape == self.search_space.output_shape
    
    def test_update_architecture_parameters(self):
        """Test updating architecture parameters with gradients."""
        import numpy as np
        
        # Get initial parameters (deep copy to avoid reference issues)
        initial_alpha = {}
        for key, value in self.darts_searcher.alpha_normal.items():
            initial_alpha[key] = np.copy(value)
        
        # Create mock gradients with larger magnitude
        gradients = {}
        for key in self.darts_searcher.alpha_normal:
            gradients[key] = np.random.normal(0, 0.1, self.darts_searcher.alpha_normal[key].shape)
        
        # Update parameters
        self.darts_searcher.update_architecture_parameters(gradients)
        
        # Verify parameters were updated
        for key in initial_alpha:
            assert not np.array_equal(initial_alpha[key], self.darts_searcher.alpha_normal[key])


class TestEvolutionaryNASSearcher:
    """Test EvolutionaryNASSearcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from automl_framework.services.nas_service import EvolutionaryNASSearcher
        
        self.layer_specs = [
            LayerSpec(LayerType.DENSE, {'units': (32, 512)}),
            LayerSpec(LayerType.CONV2D, {'filters': (16, 256)}),
            LayerSpec(LayerType.DROPOUT, {'rate': (0.1, 0.5)})
        ]
        self.search_space = SearchSpace(
            layer_specs=self.layer_specs,
            max_layers=10,
            min_layers=2,
            input_shape=(32, 32, 3),
            output_shape=(10,),
            task_type=TaskType.CLASSIFICATION
        )
        self.evolutionary_searcher = EvolutionaryNASSearcher(self.search_space)
    
    def test_evolutionary_searcher_initialization(self):
        """Test evolutionary searcher initialization."""
        assert self.evolutionary_searcher.search_space == self.search_space
        assert self.evolutionary_searcher.population_size == 50
        assert self.evolutionary_searcher.num_generations == 20
        assert self.evolutionary_searcher.mutation_rate == 0.3
        assert self.evolutionary_searcher.crossover_rate == 0.7
        assert self.evolutionary_searcher.tournament_size == 3
        assert self.evolutionary_searcher.elite_size == 5
        assert self.evolutionary_searcher.accuracy_weight == 0.7
        assert self.evolutionary_searcher.efficiency_weight == 0.3
    
    def test_initialize_population(self):
        """Test population initialization."""
        # Use smaller population for testing
        self.evolutionary_searcher.population_size = 10
        population = self.evolutionary_searcher._initialize_population()
        
        assert len(population) == 10
        
        for arch in population:
            assert isinstance(arch, Architecture)
            assert len(arch.layers) >= self.search_space.min_layers
            assert len(arch.layers) <= self.search_space.max_layers
            assert arch.parameter_count >= 0
            assert arch.flops >= 0
    
    def test_evaluate_population(self):
        """Test population evaluation."""
        # Create small test population
        population = []
        for _ in range(3):
            arch = self.evolutionary_searcher.encoder._generate_random_architecture()
            arch.parameter_count = self.evolutionary_searcher.parameter_counter.count_parameters(arch)
            arch.flops = self.evolutionary_searcher.parameter_counter.count_flops(arch)
            population.append(arch)
        
        fitness_scores = self.evolutionary_searcher._evaluate_population(population)
        
        assert len(fitness_scores) == len(population)
        assert all(isinstance(score, float) for score in fitness_scores)
        assert all(0.0 <= score <= 1.0 for score in fitness_scores)
    
    def test_calculate_fitness(self):
        """Test fitness calculation."""
        arch = self.evolutionary_searcher.encoder._generate_random_architecture()
        arch.parameter_count = 1000
        arch.flops = 10000
        
        fitness = self.evolutionary_searcher._calculate_fitness(arch)
        
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0
    
    def test_estimate_accuracy(self):
        """Test accuracy estimation."""
        # Create architecture with good characteristics
        layers = [
            Layer(LayerType.CONV2D, {'filters': 32, 'kernel_size': (3, 3)}),
            Layer(LayerType.BATCH_NORM, {}),
            Layer(LayerType.POOLING, {'pool_type': 'max'}),
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DROPOUT, {'rate': 0.3}),
            Layer(LayerType.DENSE, {'units': 10})
        ]
        connections = [Connection(i, i+1) for i in range(len(layers)-1)]
        connections.append(Connection(0, 3, "skip"))  # Skip connection
        
        arch = Architecture(
            id="test-arch",
            layers=layers,
            connections=connections,
            input_shape=(32, 32, 3),
            output_shape=(10,),
            task_type=TaskType.CLASSIFICATION
        )
        
        accuracy_score = self.evolutionary_searcher._estimate_accuracy(arch)
        
        assert isinstance(accuracy_score, float)
        assert 0.0 <= accuracy_score <= 1.0
        # Should get a good score for this well-designed architecture
        assert accuracy_score > 0.5
    
    def test_calculate_efficiency(self):
        """Test efficiency calculation."""
        arch = Architecture(
            id="test-arch",
            layers=[Layer(LayerType.DENSE, {'units': 128})],
            connections=[],
            parameter_count=1000,
            flops=10000
        )
        
        efficiency_score = self.evolutionary_searcher._calculate_efficiency(arch)
        
        assert isinstance(efficiency_score, float)
        assert 0.0 <= efficiency_score <= 1.0
    
    def test_selection(self):
        """Test parent selection."""
        # Create test population
        population = []
        fitness_scores = []
        for i in range(5):
            arch = self.evolutionary_searcher.encoder._generate_random_architecture()
            population.append(arch)
            fitness_scores.append(0.1 * i)  # Increasing fitness
        
        # Use smaller crossover rate and population size for testing
        original_population_size = self.evolutionary_searcher.population_size
        self.evolutionary_searcher.population_size = len(population)
        self.evolutionary_searcher.crossover_rate = 0.6
        
        parents = self.evolutionary_searcher._selection(population, fitness_scores)
        
        # Restore original population size
        self.evolutionary_searcher.population_size = original_population_size
        
        assert len(parents) == int(len(population) * 0.6)
        assert all(isinstance(parent, Architecture) for parent in parents)
    
    def test_crossover(self):
        """Test crossover operation."""
        # Create two parent architectures
        parent1_layers = [
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DROPOUT, {'rate': 0.3}),
            Layer(LayerType.DENSE, {'units': 64})
        ]
        parent1 = Architecture(
            id="parent1",
            layers=parent1_layers,
            connections=[Connection(0, 1), Connection(1, 2)],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        parent2_layers = [
            Layer(LayerType.DENSE, {'units': 256}),
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DROPOUT, {'rate': 0.5})
        ]
        parent2 = Architecture(
            id="parent2",
            layers=parent2_layers,
            connections=[Connection(0, 1), Connection(1, 2)],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        child1, child2 = self.evolutionary_searcher._crossover(parent1, parent2)
        
        assert isinstance(child1, Architecture)
        assert isinstance(child2, Architecture)
        assert child1.id != parent1.id
        assert child2.id != parent2.id
        assert len(child1.layers) > 0
        assert len(child2.layers) > 0
    
    def test_mutate(self):
        """Test mutation operation."""
        # Create test architecture
        layers = [
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DROPOUT, {'rate': 0.3}),
            Layer(LayerType.DENSE, {'units': 64})
        ]
        original = Architecture(
            id="original",
            layers=layers,
            connections=[Connection(0, 1), Connection(1, 2)],
            input_shape=(784,),
            output_shape=(10,)
        )
        original.parameter_count = 1000
        original.flops = 5000
        
        mutated = self.evolutionary_searcher._mutate(original)
        
        assert isinstance(mutated, Architecture)
        assert mutated.id != original.id
        assert len(mutated.layers) >= self.search_space.min_layers
        assert len(mutated.layers) <= self.search_space.max_layers
    
    def test_add_layer_mutation(self):
        """Test add layer mutation."""
        layers = [
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DENSE, {'units': 64})
        ]
        arch = Architecture(
            id="test",
            layers=deepcopy(layers),  # Use deepcopy to avoid mutation
            connections=[Connection(0, 1)],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        original_layer_count = len(arch.layers)
        mutated = self.evolutionary_searcher._add_layer_mutation(deepcopy(arch))
        
        assert len(mutated.layers) == original_layer_count + 1
        assert len(mutated.connections) >= len(mutated.layers) - 1
    
    def test_remove_layer_mutation(self):
        """Test remove layer mutation."""
        layers = [
            Layer(LayerType.DENSE, {'units': 256}),
            Layer(LayerType.DROPOUT, {'rate': 0.3}),
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DENSE, {'units': 64})
        ]
        arch = Architecture(
            id="test",
            layers=deepcopy(layers),  # Use deepcopy to avoid mutation
            connections=[Connection(i, i+1) for i in range(len(layers)-1)],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        original_layer_count = len(arch.layers)
        mutated = self.evolutionary_searcher._remove_layer_mutation(deepcopy(arch))
        
        assert len(mutated.layers) == original_layer_count - 1
        assert len(mutated.connections) >= len(mutated.layers) - 1
    
    def test_modify_layer_mutation(self):
        """Test modify layer mutation."""
        layers = [
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DROPOUT, {'rate': 0.3})
        ]
        arch = Architecture(
            id="test",
            layers=deepcopy(layers),  # Use deepcopy to avoid mutation
            connections=[Connection(0, 1)],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        original_layer_count = len(arch.layers)
        mutated = self.evolutionary_searcher._modify_layer_mutation(deepcopy(arch))
        
        # Parameters might have changed
        assert isinstance(mutated, Architecture)
        assert len(mutated.layers) == original_layer_count
    
    def test_add_skip_connection_mutation(self):
        """Test add skip connection mutation."""
        layers = [
            Layer(LayerType.DENSE, {'units': 256}),
            Layer(LayerType.DENSE, {'units': 128}),
            Layer(LayerType.DENSE, {'units': 64}),
            Layer(LayerType.DENSE, {'units': 32})
        ]
        arch = Architecture(
            id="test",
            layers=layers,
            connections=[Connection(i, i+1) for i in range(len(layers)-1)],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        original_connections = len(arch.connections)
        mutated = self.evolutionary_searcher._add_skip_connection_mutation(arch)
        
        # Might have added a skip connection
        assert len(mutated.connections) >= original_connections
    
    def test_survival_selection(self):
        """Test survival selection."""
        # Create test population with known fitness scores
        population = []
        fitness_scores = []
        for i in range(10):
            arch = self.evolutionary_searcher.encoder._generate_random_architecture()
            population.append(arch)
            fitness_scores.append(i * 0.1)  # Fitness from 0.0 to 0.9
        
        # Use smaller population size for testing
        self.evolutionary_searcher.population_size = 5
        survivors = self.evolutionary_searcher._survival_selection(population, fitness_scores)
        
        assert len(survivors) == 5
        assert all(isinstance(survivor, Architecture) for survivor in survivors)
    
    def test_search_architectures(self):
        """Test complete evolutionary search."""
        # Use smaller parameters for faster testing
        self.evolutionary_searcher.population_size = 10
        self.evolutionary_searcher.num_generations = 3
        
        architectures = self.evolutionary_searcher.search_architectures(num_architectures=3)
        
        assert isinstance(architectures, list)
        assert len(architectures) <= 3
        
        for arch in architectures:
            assert isinstance(arch, Architecture)
            assert len(arch.layers) >= self.search_space.min_layers
            assert len(arch.layers) <= self.search_space.max_layers
            assert arch.parameter_count >= 0
            assert arch.flops >= 0


class TestNASService:
    """Test NASService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nas_service = NASService()
    
    def test_define_search_space_image_classification(self):
        """Test defining search space for image classification."""
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.IMAGE
        )
        
        # Verify search space properties
        assert isinstance(search_space, SearchSpace)
        assert search_space.task_type == TaskType.CLASSIFICATION
        assert search_space.max_layers > 0
        assert search_space.min_layers > 0
        
        # Verify image-specific layers are included
        layer_types = search_space.get_layer_types()
        assert LayerType.CONV2D in layer_types
        assert LayerType.POOLING in layer_types
        assert LayerType.BATCH_NORM in layer_types
        assert LayerType.DENSE in layer_types
        assert LayerType.DROPOUT in layer_types
    
    def test_define_search_space_text_classification(self):
        """Test defining search space for text classification."""
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.TEXT
        )
        
        # Verify search space properties
        assert isinstance(search_space, SearchSpace)
        assert search_space.task_type == TaskType.CLASSIFICATION
        
        # Verify text-specific layers are included
        layer_types = search_space.get_layer_types()
        assert LayerType.LSTM in layer_types
        assert LayerType.GRU in layer_types
        assert LayerType.DENSE in layer_types
        assert LayerType.DROPOUT in layer_types
    
    def test_define_search_space_tabular_data(self):
        """Test defining search space for tabular data."""
        search_space = self.nas_service.define_search_space(
            TaskType.REGRESSION,
            DataType.TABULAR
        )
        
        # Verify search space properties
        assert isinstance(search_space, SearchSpace)
        assert search_space.task_type == TaskType.REGRESSION
        
        # Verify tabular-specific layers are included
        layer_types = search_space.get_layer_types()
        assert LayerType.DENSE in layer_types
        assert LayerType.DROPOUT in layer_types
    
    def test_search_architectures_darts(self):
        """Test architecture search using DARTS method."""
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.IMAGE
        )
        
        architectures = self.nas_service.search_architectures(
            search_space, None, method='darts'
        )
        
        # Verify returned architectures
        assert isinstance(architectures, list)
        assert len(architectures) > 0
        
        for arch in architectures:
            assert isinstance(arch, Architecture)
            assert arch.parameter_count >= 0
            assert arch.flops >= 0
    
    def test_search_architectures_evolutionary(self):
        """Test architecture search using evolutionary method."""
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.IMAGE
        )
        
        architectures = self.nas_service.search_architectures(
            search_space, None, method='evolutionary'
        )
        
        # Verify returned architectures
        assert isinstance(architectures, list)
        assert len(architectures) > 0
        
        for arch in architectures:
            assert isinstance(arch, Architecture)
            assert arch.parameter_count >= 0
            assert arch.flops >= 0
    
    def test_search_architectures_random(self):
        """Test architecture search using random method."""
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.IMAGE
        )
        
        architectures = self.nas_service.search_architectures(
            search_space, None, method='random'
        )
        
        # Verify returned architectures
        assert isinstance(architectures, list)
        assert len(architectures) > 0
        
        for arch in architectures:
            assert isinstance(arch, Architecture)
            assert arch.parameter_count >= 0
            assert arch.flops >= 0
    
    def test_search_architectures_invalid_method(self):
        """Test architecture search with invalid method."""
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.IMAGE
        )
        
        with pytest.raises(ValueError, match="Unknown search method"):
            self.nas_service.search_architectures(
                search_space, None, method='invalid_method'
            )
    
    def test_evaluate_architecture_placeholder(self):
        """Test architecture evaluation (placeholder implementation)."""
        # Create a simple architecture
        layers = [Layer(LayerType.DENSE, {'units': 128})]
        architecture = Architecture(
            id="test-arch",
            layers=layers,
            connections=[],
            input_shape=(784,),
            output_shape=(10,)
        )
        
        metrics = self.nas_service.evaluate_architecture(architecture, None)
        
        # Verify returned metrics
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.accuracy is not None
        assert metrics.loss is not None
        assert metrics.training_time is not None
        assert metrics.inference_time is not None
        assert 0 <= metrics.accuracy <= 1
        assert metrics.loss >= 0
        assert metrics.training_time > 0
        assert metrics.inference_time > 0
    
    def test_rank_architectures(self):
        """Test ranking architectures by performance."""
        # Create test architectures
        arch1 = Architecture(
            id="arch1",
            layers=[Layer(LayerType.DENSE, {'units': 64})],
            connections=[],
            input_shape=(784,),
            output_shape=(10,),
            parameter_count=1000,
            flops=5000
        )
        
        arch2 = Architecture(
            id="arch2",
            layers=[Layer(LayerType.DENSE, {'units': 128})],
            connections=[],
            input_shape=(784,),
            output_shape=(10,),
            parameter_count=2000,
            flops=10000
        )
        
        architectures = [arch1, arch2]
        
        # Mock the evaluate_architecture method to return predictable results
        def mock_evaluate(arch, dataset_metadata):
            if arch.id == "arch1":
                return PerformanceMetrics(accuracy=0.9, loss=0.1, training_time=100, inference_time=5)
            else:
                return PerformanceMetrics(accuracy=0.85, loss=0.15, training_time=200, inference_time=10)
        
        self.nas_service.evaluate_architecture = mock_evaluate
        
        ranked_architectures = self.nas_service.rank_architectures(architectures, None)
        
        # Verify ranking
        assert isinstance(ranked_architectures, list)
        assert len(ranked_architectures) == 2
        
        # Each item should be a tuple of (architecture, score)
        for arch, score in ranked_architectures:
            assert isinstance(arch, Architecture)
            assert isinstance(score, float)
            assert 0 <= score <= 1
        
        # Verify architectures are sorted by score (descending)
        scores = [score for arch, score in ranked_architectures]
        assert scores == sorted(scores, reverse=True)
    
    def test_get_best_architectures(self):
        """Test getting best architectures."""
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.IMAGE
        )
        
        best_architectures = self.nas_service.get_best_architectures(
            search_space, None, method='random', top_k=3
        )
        
        # Verify returned architectures
        assert isinstance(best_architectures, list)
        assert len(best_architectures) <= 3
        
        for arch in best_architectures:
            assert isinstance(arch, Architecture)
            assert arch.parameter_count >= 0
            assert arch.flops >= 0
    
    def test_update_architecture_metrics(self):
        """Test updating architecture with parameter count and FLOPs."""
        layers = [
            Layer(LayerType.DENSE, {'input_size': 784, 'units': 128, 'use_bias': True}),
            Layer(LayerType.DENSE, {'input_size': 128, 'units': 10, 'use_bias': True})
        ]
        
        architecture = Architecture(
            id="test-arch",
            layers=layers,
            connections=[Connection(0, 1)],
            input_shape=(784,),
            output_shape=(10,),
            parameter_count=0,  # Will be updated
            flops=0  # Will be updated
        )
        
        updated_arch = self.nas_service._update_architecture_metrics(architecture)
        
        # Verify metrics were updated
        assert updated_arch.parameter_count > 0
        assert updated_arch.flops >= 0
        
        # Verify specific parameter count
        expected_params = (784 * 128 + 128) + (128 * 10 + 10)
        assert updated_arch.parameter_count == expected_params


if __name__ == "__main__":
    pytest.main([__file__])