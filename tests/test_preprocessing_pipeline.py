"""
Unit tests for preprocessing pipeline framework.

Tests the preprocessing transformers, pipeline, and auto preprocessor functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from automl_framework.services.data_processing import (
    MissingValueImputer, NumericalScaler, CategoricalEncoder,
    PreprocessingPipeline, AutoPreprocessor, DataProcessingService
)
from automl_framework.models.data_models import DataType, TaskType
from automl_framework.core.interfaces import DatasetMetadata
from automl_framework.models.data_models import Feature


class TestMissingValueImputer:
    """Test suite for MissingValueImputer."""
    
    def test_init(self):
        """Test imputer initialization."""
        imputer = MissingValueImputer()
        assert imputer.name == 'missing_value_imputer'
        assert imputer.strategy == 'auto'
        assert not imputer.fitted
    
    def test_fit_transform_numeric(self):
        """Test imputation of numeric data."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, np.nan, 4.0, 5.0]
        })
        
        imputer = MissingValueImputer(strategy='median')
        result = imputer.fit_transform(df)
        
        assert not result['numeric'].isnull().any()
        assert result['numeric'].iloc[2] == 3.0  # median of [1,2,4,5]
    
    def test_fit_transform_categorical(self):
        """Test imputation of categorical data."""
        df = pd.DataFrame({
            'categorical': ['A', 'B', None, 'A', 'A']
        })
        
        imputer = MissingValueImputer(strategy='most_frequent')
        result = imputer.fit_transform(df)
        
        assert not result['categorical'].isnull().any()
        assert result['categorical'].iloc[2] == 'A'  # most frequent
    
    def test_auto_strategy(self):
        """Test automatic strategy selection."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical': ['A', 'B', None, 'A', 'A']
        })
        
        imputer = MissingValueImputer(strategy='auto')
        result = imputer.fit_transform(df)
        
        assert not result.isnull().any().any()
        assert result['numeric'].iloc[2] == 3.0  # median for numeric
        assert result['categorical'].iloc[2] == 'A'  # mode for categorical


class TestNumericalScaler:
    """Test suite for NumericalScaler."""
    
    def test_init(self):
        """Test scaler initialization."""
        scaler = NumericalScaler()
        assert scaler.name == 'numerical_scaler'
        assert scaler.method == 'standard'
        assert not scaler.fitted
    
    def test_standard_scaling(self):
        """Test standard scaling."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0, 4.0, 5.0],
            'categorical': ['A', 'B', 'C', 'D', 'E']
        })
        
        scaler = NumericalScaler(method='standard')
        result = scaler.fit_transform(df)
        
        # Check that numeric column is scaled (mean ~0, std close to 1)
        assert abs(result['numeric'].mean()) < 1e-10
        assert abs(result['numeric'].std(ddof=0) - 1.0) < 1e-10
        
        # Check that categorical column is unchanged
        assert result['categorical'].equals(df['categorical'])
    
    def test_minmax_scaling(self):
        """Test min-max scaling."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        scaler = NumericalScaler(method='minmax')
        result = scaler.fit_transform(df)
        
        # Check that values are in [0, 1] range
        assert result['numeric'].min() == 0.0
        assert result['numeric'].max() == 1.0


class TestCategoricalEncoder:
    """Test suite for CategoricalEncoder."""
    
    def test_init(self):
        """Test encoder initialization."""
        encoder = CategoricalEncoder()
        assert encoder.name == 'categorical_encoder'
        assert encoder.method == 'auto'
        assert not encoder.fitted
    
    def test_label_encoding(self):
        """Test label encoding."""
        df = pd.DataFrame({
            'categorical': ['A', 'B', 'C', 'A', 'B'],
            'numeric': [1, 2, 3, 4, 5]
        })
        
        encoder = CategoricalEncoder(method='label')
        result = encoder.fit_transform(df)
        
        # Check that categorical column is encoded as integers
        assert pd.api.types.is_numeric_dtype(result['categorical'])
        assert set(result['categorical'].unique()) == {0, 1, 2}  # A, B, C encoded
        
        # Check that numeric column is unchanged
        assert result['numeric'].equals(df['numeric'])
    
    def test_onehot_encoding(self):
        """Test one-hot encoding."""
        df = pd.DataFrame({
            'categorical': ['A', 'B', 'C', 'A', 'B']
        })
        
        encoder = CategoricalEncoder(method='onehot')
        result = encoder.fit_transform(df)
        
        # Check that we have 3 new columns (A, B, C)
        expected_columns = ['categorical_A', 'categorical_B', 'categorical_C']
        assert all(col in result.columns for col in expected_columns)
        assert 'categorical' not in result.columns  # Original column removed
        
        # Check that encoding is correct
        assert result.loc[0, 'categorical_A'] == 1.0
        assert result.loc[0, 'categorical_B'] == 0.0
    
    def test_auto_encoding(self):
        """Test automatic encoding method selection."""
        # Few categories - should use one-hot
        df_few = pd.DataFrame({
            'categorical': ['A', 'B', 'C'] * 10
        })
        
        encoder_few = CategoricalEncoder(method='auto', max_categories=5)
        result_few = encoder_few.fit_transform(df_few)
        
        # Should have one-hot encoded columns
        assert 'categorical_A' in result_few.columns
        
        # Many categories - should use label encoding
        df_many = pd.DataFrame({
            'categorical': [f'cat_{i}' for i in range(20)] * 2
        })
        
        encoder_many = CategoricalEncoder(method='auto', max_categories=5)
        result_many = encoder_many.fit_transform(df_many)
        
        # Should have label encoded column
        assert 'categorical' in result_many.columns
        assert pd.api.types.is_numeric_dtype(result_many['categorical'])


class TestPreprocessingPipeline:
    """Test suite for PreprocessingPipeline."""
    
    def test_init(self):
        """Test pipeline initialization."""
        pipeline = PreprocessingPipeline()
        assert len(pipeline.transformers) == 0
        assert not pipeline.fitted
    
    def test_add_transformer(self):
        """Test adding transformers to pipeline."""
        pipeline = PreprocessingPipeline()
        imputer = MissingValueImputer()
        scaler = NumericalScaler()
        
        pipeline.add_transformer(imputer)
        pipeline.add_transformer(scaler)
        
        assert len(pipeline.transformers) == 2
        assert pipeline.transformers[0] == imputer
        assert pipeline.transformers[1] == scaler
    
    def test_fit_transform_pipeline(self):
        """Test complete pipeline fit and transform."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical': ['A', 'B', None, 'A', 'B']
        })
        
        pipeline = PreprocessingPipeline()
        pipeline.add_transformer(MissingValueImputer(strategy='auto'))
        pipeline.add_transformer(CategoricalEncoder(method='label'))
        pipeline.add_transformer(NumericalScaler(method='standard'))
        
        result = pipeline.fit_transform(df)
        
        # Check that all transformations were applied
        assert not result.isnull().any().any()  # No missing values
        assert pd.api.types.is_numeric_dtype(result['categorical'])  # Encoded
        assert abs(result['numeric'].mean()) < 1e-10  # Scaled
    
    def test_pipeline_serialization(self):
        """Test pipeline serialization to dictionary."""
        pipeline = PreprocessingPipeline()
        pipeline.add_transformer(MissingValueImputer())
        pipeline.add_transformer(NumericalScaler())
        
        config = pipeline.to_dict()
        
        assert 'transformers' in config
        assert 'fitted' in config
        assert len(config['transformers']) == 2


class TestAutoPreprocessor:
    """Test suite for AutoPreprocessor."""
    
    def test_init(self):
        """Test auto preprocessor initialization."""
        preprocessor = AutoPreprocessor()
        assert preprocessor.test_size == 0.2
        assert preprocessor.validation_size == 0.2
        assert preprocessor.random_state == 42
    
    def test_create_preprocessing_pipeline(self):
        """Test automatic pipeline creation based on metadata."""
        # Create mock metadata
        features = [
            Feature(name='numeric', data_type='float64', is_categorical=False, 
                   missing_percentage=10.0, unique_values=50),
            Feature(name='categorical', data_type='object', is_categorical=True, 
                   missing_percentage=5.0, unique_values=3)
        ]
        
        metadata = DatasetMetadata(
            id='test',
            name='test',
            data_type=DataType.TABULAR,
            task_type=TaskType.CLASSIFICATION,
            size=100,
            features=features,
            target_column='target',
            class_distribution={'A': 50, 'B': 50},
            statistics={}
        )
        
        preprocessor = AutoPreprocessor()
        pipeline = preprocessor.create_preprocessing_pipeline(metadata)
        
        # Should have imputer, encoder, and scaler
        assert len(pipeline.transformers) == 3
        assert isinstance(pipeline.transformers[0], MissingValueImputer)
        assert isinstance(pipeline.transformers[1], CategoricalEncoder)
        assert isinstance(pipeline.transformers[2], NumericalScaler)
    
    def test_apply_preprocessing(self):
        """Test applying preprocessing with data splitting."""
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': ['A', 'B'] * 50,
            'target': ['class1', 'class2'] * 50
        })
        
        pipeline = PreprocessingPipeline()
        pipeline.add_transformer(CategoricalEncoder(method='label'))
        pipeline.add_transformer(NumericalScaler(method='standard'))
        
        preprocessor = AutoPreprocessor(test_size=0.2, validation_size=0.2)
        processed_data = preprocessor.apply_preprocessing(pipeline, df, target_column='target')
        
        # Check data splits (approximately correct due to rounding)
        total_size = len(processed_data.train_data) + len(processed_data.validation_data) + len(processed_data.test_data)
        assert total_size == 100  # Should equal original dataset size
        assert len(processed_data.test_data) == 20  # 20% test
        assert len(processed_data.train_data) + len(processed_data.validation_data) == 80  # 80% train+val
        
        # Check that target column is present in all splits
        assert 'target' in processed_data.train_data.columns
        assert 'target' in processed_data.validation_data.columns
        assert 'target' in processed_data.test_data.columns
        
        # Check that preprocessing was applied
        assert processed_data.preprocessing_pipeline is not None
        assert len(processed_data.feature_names) > 0


class TestDataProcessingService:
    """Test suite for DataProcessingService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataProcessingService()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test service initialization."""
        service = DataProcessingService()
        assert service.analyzer is not None
        assert service.preprocessor is not None
    
    def test_process_dataset_complete(self):
        """Test complete dataset processing workflow."""
        # Create test dataset
        df = pd.DataFrame({
            'feature1': [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'target': ['class1', 'class2', 'class1', 'class2', 'class1', 
                      'class2', 'class1', 'class2', 'class1', 'class2']
        })
        
        csv_path = os.path.join(self.temp_dir, "test_dataset.csv")
        df.to_csv(csv_path, index=False)
        
        # Process dataset
        metadata, processed_data = self.service.process_dataset(csv_path, target_column='target')
        
        # Check metadata
        assert metadata.data_type == DataType.TABULAR
        assert metadata.size == 10
        assert len(metadata.features) == 4
        
        # Check processed data
        assert processed_data.train_data is not None
        assert processed_data.validation_data is not None
        assert processed_data.test_data is not None
        assert processed_data.preprocessing_pipeline is not None
        
        # Check that missing values were handled (target column might still have some)
        # Focus on feature columns only
        feature_cols = [col for col in processed_data.train_data.columns if col != 'target']
        train_missing = processed_data.train_data[feature_cols].isnull().sum().sum()
        val_missing = processed_data.validation_data[feature_cols].isnull().sum().sum()
        test_missing = processed_data.test_data[feature_cols].isnull().sum().sum()
        total_feature_missing = train_missing + val_missing + test_missing
        assert total_feature_missing == 0
    
    def test_interface_compliance(self):
        """Test that service implements IDataProcessor interface correctly."""
        # Test analyze_dataset method
        df = pd.DataFrame({'feature': [1, 2, 3], 'target': ['A', 'B', 'A']})
        csv_path = os.path.join(self.temp_dir, "test.csv")
        df.to_csv(csv_path, index=False)
        
        metadata = self.service.analyze_dataset(csv_path)
        assert metadata is not None
        
        # Test create_preprocessing_pipeline method
        pipeline = self.service.create_preprocessing_pipeline(metadata)
        assert pipeline is not None
        
        # Test apply_preprocessing method
        processed_data = self.service.apply_preprocessing(pipeline, df)
        assert processed_data is not None


class TestPreprocessingEdgeCases:
    """Test edge cases and error handling in preprocessing."""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframes."""
        df = pd.DataFrame()
        
        imputer = MissingValueImputer()
        result = imputer.fit_transform(df)
        assert len(result) == 0
    
    def test_all_missing_column(self):
        """Test handling of columns with all missing values."""
        df = pd.DataFrame({
            'all_missing': [np.nan, np.nan, np.nan],
            'some_data': [1, 2, 3]
        })
        
        imputer = MissingValueImputer(strategy='median')
        result = imputer.fit_transform(df)
        
        # All missing column should be imputed with median of other values or 0
        assert not result['all_missing'].isnull().any()
    
    def test_single_category(self):
        """Test encoding of categorical column with single category."""
        df = pd.DataFrame({
            'single_cat': ['A', 'A', 'A', 'A']
        })
        
        encoder = CategoricalEncoder(method='label')
        result = encoder.fit_transform(df)
        
        # Should still work, all values should be 0
        assert all(result['single_cat'] == 0)
    
    def test_constant_numeric_column(self):
        """Test scaling of constant numeric column."""
        df = pd.DataFrame({
            'constant': [5.0, 5.0, 5.0, 5.0]
        })
        
        scaler = NumericalScaler(method='standard')
        result = scaler.fit_transform(df)
        
        # Constant column should remain constant (or become 0 after scaling)
        assert result['constant'].nunique() == 1
    
    def test_pipeline_with_no_transformers(self):
        """Test pipeline with no transformers."""
        df = pd.DataFrame({'feature': [1, 2, 3]})
        
        pipeline = PreprocessingPipeline()
        result = pipeline.fit_transform(df)
        
        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, df)
    
    def test_transform_before_fit(self):
        """Test error when transforming before fitting."""
        df = pd.DataFrame({'feature': [1, 2, 3]})
        
        imputer = MissingValueImputer()
        
        with pytest.raises(ValueError, match="must be fitted before transform"):
            imputer.transform(df)