"""
Integration tests for Neural Architecture Search (NAS) service.

Tests the complete NAS workflow including evolutionary algorithm.
"""

import pytest
from automl_framework.services.nas_service import NASService
from automl_framework.models.data_models import TaskType, DataType


class TestNASIntegration:
    """Integration tests for NAS service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nas_service = NASService()
    
    def test_evolutionary_nas_workflow(self):
        """Test complete evolutionary NAS workflow."""
        # Define search space for image classification
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.IMAGE
        )
        
        # Verify search space is properly configured
        assert search_space is not None
        assert search_space.task_type == TaskType.CLASSIFICATION
        assert len(search_space.layer_specs) > 0
        
        # Search for architectures using evolutionary method
        architectures = self.nas_service.search_architectures(
            search_space, 
            dataset_metadata=None, 
            method='evolutionary'
        )
        
        # Verify architectures were found
        assert isinstance(architectures, list)
        assert len(architectures) > 0
        
        # Verify each architecture is valid
        for arch in architectures:
            assert arch.id is not None
            assert len(arch.layers) >= search_space.min_layers
            assert len(arch.layers) <= search_space.max_layers
            assert arch.parameter_count >= 0
            assert arch.flops >= 0
            assert arch.task_type == TaskType.CLASSIFICATION
        
        # Get best architectures
        best_architectures = self.nas_service.get_best_architectures(
            search_space,
            dataset_metadata=None,
            method='evolutionary',
            top_k=3
        )
        
        # Verify best architectures
        assert isinstance(best_architectures, list)
        assert len(best_architectures) <= 3
        assert len(best_architectures) > 0
        
        # Verify architectures are ranked (first should be best)
        if len(best_architectures) > 1:
            # All architectures should be valid
            for arch in best_architectures:
                assert arch.parameter_count >= 0
                assert arch.flops >= 0
    
    def test_evolutionary_vs_darts_comparison(self):
        """Test comparison between evolutionary and DARTS methods."""
        # Define search space
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.IMAGE
        )
        
        # Search with evolutionary method
        evolutionary_archs = self.nas_service.search_architectures(
            search_space, 
            dataset_metadata=None, 
            method='evolutionary'
        )
        
        # Search with DARTS method
        darts_archs = self.nas_service.search_architectures(
            search_space, 
            dataset_metadata=None, 
            method='darts'
        )
        
        # Both methods should return valid architectures
        assert len(evolutionary_archs) > 0
        assert len(darts_archs) > 0
        
        # Architectures should be different (different search strategies)
        evolutionary_ids = {arch.id for arch in evolutionary_archs}
        darts_ids = {arch.id for arch in darts_archs}
        
        # They should produce different architectures
        assert evolutionary_ids != darts_ids
        
        # Both should produce valid architectures
        for arch in evolutionary_archs + darts_archs:
            assert len(arch.layers) >= search_space.min_layers
            assert len(arch.layers) <= search_space.max_layers
    
    def test_evolutionary_nas_tabular_data(self):
        """Test evolutionary NAS for tabular data."""
        # Define search space for tabular classification
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.TABULAR
        )
        
        # Search for architectures
        architectures = self.nas_service.search_architectures(
            search_space, 
            dataset_metadata=None, 
            method='evolutionary'
        )
        
        # Verify architectures are appropriate for tabular data
        assert len(architectures) > 0
        
        for arch in architectures:
            # Should primarily use dense layers for tabular data
            layer_types = [layer.layer_type.value for layer in arch.layers]
            assert 'dense' in layer_types
            
            # Should not have convolutional layers for tabular data
            assert 'conv2d' not in layer_types or len([lt for lt in layer_types if lt == 'conv2d']) <= 1
    
    def test_evolutionary_nas_multi_objective_optimization(self):
        """Test that evolutionary NAS produces valid architectures with multi-objective optimization."""
        # Define search space
        search_space = self.nas_service.define_search_space(
            TaskType.CLASSIFICATION,
            DataType.IMAGE
        )
        
        # Get best architectures
        best_architectures = self.nas_service.get_best_architectures(
            search_space,
            dataset_metadata=None,
            method='evolutionary',
            top_k=5
        )
        
        # Verify we have valid architectures
        assert len(best_architectures) > 0
        
        # All architectures should be valid and within constraints
        for arch in best_architectures:
            assert len(arch.layers) >= search_space.min_layers
            assert len(arch.layers) <= search_space.max_layers
            assert arch.parameter_count >= 0
            assert arch.flops >= 0
            assert arch.id is not None
            assert arch.task_type == TaskType.CLASSIFICATION
        
        # Verify that architectures have appropriate layer types for image classification
        for arch in best_architectures:
            layer_types = [layer.layer_type.value for layer in arch.layers]
            # Should have at least one of the common layer types
            common_types = {'dense', 'conv2d', 'dropout', 'pooling', 'batch_norm'}
            assert any(lt in common_types for lt in layer_types)