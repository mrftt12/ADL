"""
Comprehensive unit tests for data processing service.

Tests cover dataset analysis, preprocessing pipeline, feature engineering,
and data validation with various data types and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.preprocessing import StandardScaler, LabelEncoder

from automl_framework.services.data_processing import (
    DataProcessingService, DatasetAnalyzer, PreprocessingPipeline,
    FeatureEngineer, DataValidator
)
from automl_framework.models.data_models import (
    Dataset, Feature, DataType, TaskType
)
from tests.test_utils import (
    MockDatasetGenerator, TestDataManager, assert_metrics_valid,
    mock_tabular_dataset, mock_image_dataset, test_data_manager
)


class TestDatasetAnalyzer:
    """Test DatasetAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DatasetAnalyzer()
    
    def test_analyze_tabular_dataset(self, mock_tabular_dataset):
        """Test analyzing tabular dataset."""
        df, temp_file = mock_tabular_dataset
        
        metadata = self.analyzer.analyze_dataset(temp_file)
        
        assert metadata['data_type'] == DataType.TABULAR
        assert metadata['n_samples'] == len(df)
        assert metadata['n_features'] == len(df.columns) - 1  # Excluding target
        assert 'feature_types' in metadata
        assert 'missing_values' in metadata
        assert 'target_column' in metadata
    
    def test_detect_data_type_tabular(self, mock_tabular_dataset):
        """Test data type detection for tabular data."""
        df, temp_file = mock_tabular_dataset
        
        data_type = self.analyzer._detect_data_type(temp_file)
        assert data_type == DataType.TABULAR
    
    def test_detect_data_type_image_directory(self, test_data_manager):
        """Test data type detection for image directory."""
        images, labels, temp_dir = test_data_manager.create_image_dataset()
        
        data_type = self.analyzer._detect_data_type(temp_dir)
        assert data_type == DataType.IMAGE
    
    def test_analyze_feature_types(self, mock_tabular_dataset):
        """Test feature type analysis."""
        df, temp_file = mock_tabular_dataset
        
        feature_types = self.analyzer._analyze_feature_types(df)
        
        assert isinstance(feature_types, dict)
        assert len(feature_types) == len(df.columns)
        
        # Check that numeric features are detected
        numeric_features = [col for col in df.columns if col.startswith('numeric_')]
        for col in numeric_features:
            assert feature_types[col] in ['numeric', 'integer']
        
        # Check that categorical features are detected
        categorical_features = [col for col in df.columns if col.startswith('categorical_')]
        for col in categorical_features:
            assert feature_types[col] == 'categorical'
    
    def test_calculate_missing_values(self, mock_tabular_dataset):
        """Test missing value calculation."""
        df, temp_file = mock_tabular_dataset
        
        missing_stats = self.analyzer._calculate_missing_values(df)
        
        assert isinstance(missing_stats, dict)
        assert len(missing_stats) == len(df.columns)
        
        for col, missing_pct in missing_stats.items():
            assert 0 <= missing_pct <= 100
    
    def test_detect_target_column(self, mock_tabular_dataset):
        """Test target column detection."""
        df, temp_file = mock_tabular_dataset
        
        target_col = self.analyzer._detect_target_column(df)
        assert target_col == 'target'
    
    def test_calculate_dataset_statistics(self, mock_tabular_dataset):
        """Test dataset statistics calculation."""
        df, temp_file = mock_tabular_dataset
        
        stats = self.analyzer._calculate_dataset_statistics(df)
        
        assert 'n_samples' in stats
        assert 'n_features' in stats
        assert 'memory_usage_mb' in stats
        assert stats['n_samples'] == len(df)
        assert stats['n_features'] == len(df.columns)
    
    def test_analyze_class_distribution(self, mock_tabular_dataset):
        """Test class distribution analysis."""
        df, temp_file = mock_tabular_dataset
        
        class_dist = self.analyzer._analyze_class_distribution(df, 'target')
        
        assert isinstance(class_dist, dict)
        assert 'class_counts' in class_dist
        assert 'class_balance' in class_dist
        assert 'is_balanced' in class_dist
    
    def test_analyze_feature_correlations(self, mock_tabular_dataset):
        """Test feature correlation analysis."""
        df, temp_file = mock_tabular_dataset
        
        # Select only numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_cols]
        
        correlations = self.analyzer._analyze_feature_correlations(numeric_df)
        
        assert isinstance(correlations, dict)
        assert 'correlation_matrix' in correlations
        assert 'high_correlations' in correlations
    
    def test_analyze_empty_dataset(self):
        """Test analyzing empty dataset."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Dataset is empty"):
            self.analyzer._calculate_dataset_statistics(empty_df)
    
    def test_analyze_single_column_dataset(self):
        """Test analyzing dataset with single column."""
        df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        
        stats = self.analyzer._calculate_dataset_statistics(df)
        assert stats['n_features'] == 1
        
        # Should not detect target column
        target_col = self.analyzer._detect_target_column(df)
        assert target_col is None


class TestPreprocessingPipeline:
    """Test PreprocessingPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = PreprocessingPipeline()
    
    def test_create_pipeline_for_tabular_data(self, mock_tabular_dataset):
        """Test creating preprocessing pipeline for tabular data."""
        df, temp_file = mock_tabular_dataset
        
        # Mock dataset metadata
        metadata = {
            'data_type': DataType.TABULAR,
            'feature_types': {col: 'numeric' if col.startswith('numeric') else 'categorical' 
                            for col in df.columns if col != 'target'},
            'missing_values': {col: 10.0 for col in df.columns},
            'target_column': 'target'
        }
        
        pipeline_steps = self.pipeline.create_pipeline(metadata)
        
        assert isinstance(pipeline_steps, list)
        assert len(pipeline_steps) > 0
        
        # Check that essential steps are included
        step_names = [step['name'] for step in pipeline_steps]
        assert 'missing_value_imputation' in step_names
        assert 'feature_scaling' in step_names
        assert 'categorical_encoding' in step_names
    
    def test_handle_missing_values_numeric(self):
        """Test missing value handling for numeric features."""
        df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0]
        })
        
        processed_df = self.pipeline._handle_missing_values(
            df, {'numeric_col': 'mean'}, 'target'
        )
        
        assert not processed_df['numeric_col'].isna().any()
        assert processed_df['numeric_col'].iloc[2] == df['numeric_col'].mean()
    
    def test_handle_missing_values_categorical(self):
        """Test missing value handling for categorical features."""
        df = pd.DataFrame({
            'categorical_col': ['A', 'B', None, 'A', 'C'],
            'target': [0, 1, 0, 1, 0]
        })
        
        processed_df = self.pipeline._handle_missing_values(
            df, {'categorical_col': 'mode'}, 'target'
        )
        
        assert not processed_df['categorical_col'].isna().any()
        assert processed_df['categorical_col'].iloc[2] == 'A'  # Most frequent
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        df = pd.DataFrame({
            'categorical_col': ['A', 'B', 'C', 'A', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        
        encoded_df = self.pipeline._encode_categorical_features(
            df, ['categorical_col'], 'target'
        )
        
        # Check that categorical column is encoded
        assert 'categorical_col' not in encoded_df.columns
        # Should have one-hot encoded columns
        encoded_cols = [col for col in encoded_df.columns if col.startswith('categorical_col_')]
        assert len(encoded_cols) > 0
    
    def test_scale_numeric_features(self):
        """Test numeric feature scaling."""
        df = pd.DataFrame({
            'numeric_col1': [1, 2, 3, 4, 5],
            'numeric_col2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        scaled_df, scaler = self.pipeline._scale_numeric_features(
            df, ['numeric_col1', 'numeric_col2'], 'target'
        )
        
        # Check that features are scaled (mean ~0, std ~1)
        assert abs(scaled_df['numeric_col1'].mean()) < 1e-10
        assert abs(scaled_df['numeric_col1'].std() - 1.0) < 1e-10
        assert abs(scaled_df['numeric_col2'].mean()) < 1e-10
        assert abs(scaled_df['numeric_col2'].std() - 1.0) < 1e-10
    
    def test_remove_outliers(self):
        """Test outlier removal."""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 95)
        outliers = np.array([10, -10, 15, -15, 20])
        data = np.concatenate([normal_data, outliers])
        
        df = pd.DataFrame({
            'numeric_col': data,
            'target': np.random.randint(0, 2, 100)
        })
        
        cleaned_df = self.pipeline._remove_outliers(df, ['numeric_col'], method='iqr')
        
        # Should have fewer rows after outlier removal
        assert len(cleaned_df) < len(df)
        
        # Extreme outliers should be removed
        assert cleaned_df['numeric_col'].max() < 10
        assert cleaned_df['numeric_col'].min() > -10
    
    def test_feature_selection(self):
        """Test feature selection."""
        # Create data with some irrelevant features
        np.random.seed(42)
        n_samples = 1000
        
        # Relevant features
        X_relevant = np.random.normal(0, 1, (n_samples, 3))
        # Irrelevant features (random noise)
        X_irrelevant = np.random.normal(0, 1, (n_samples, 7))
        
        # Target correlated with relevant features
        y = (X_relevant[:, 0] + X_relevant[:, 1] > 0).astype(int)
        
        df = pd.DataFrame(
            np.column_stack([X_relevant, X_irrelevant]),
            columns=[f'feature_{i}' for i in range(10)]
        )
        df['target'] = y
        
        selected_df = self.pipeline._select_features(
            df, [f'feature_{i}' for i in range(10)], 'target', k=5
        )
        
        # Should select top k features
        feature_cols = [col for col in selected_df.columns if col != 'target']
        assert len(feature_cols) == 5
    
    def test_apply_pipeline(self, mock_tabular_dataset):
        """Test applying complete preprocessing pipeline."""
        df, temp_file = mock_tabular_dataset
        
        # Create pipeline steps
        pipeline_steps = [
            {
                'name': 'missing_value_imputation',
                'function': self.pipeline._handle_missing_values,
                'params': {'strategies': {'numeric_feature_0': 'mean'}}
            },
            {
                'name': 'feature_scaling',
                'function': self.pipeline._scale_numeric_features,
                'params': {'method': 'standard'}
            }
        ]
        
        processed_df, artifacts = self.pipeline.apply_pipeline(df, pipeline_steps, 'target')
        
        assert isinstance(processed_df, pd.DataFrame)
        assert isinstance(artifacts, dict)
        assert len(processed_df) <= len(df)  # May be smaller due to preprocessing
    
    def test_pipeline_serialization(self):
        """Test pipeline serialization and deserialization."""
        pipeline_steps = [
            {
                'name': 'missing_value_imputation',
                'function': 'handle_missing_values',
                'params': {'strategies': {'col1': 'mean'}}
            }
        ]
        
        serialized = self.pipeline.serialize_pipeline(pipeline_steps)
        deserialized = self.pipeline.deserialize_pipeline(serialized)
        
        assert deserialized == pipeline_steps
    
    def test_invalid_pipeline_step(self):
        """Test handling invalid pipeline step."""
        invalid_steps = [
            {
                'name': 'invalid_step',
                'function': 'nonexistent_function',
                'params': {}
            }
        ]
        
        df = pd.DataFrame({'col1': [1, 2, 3], 'target': [0, 1, 0]})
        
        with pytest.raises(AttributeError):
            self.pipeline.apply_pipeline(df, invalid_steps, 'target')


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()
    
    def test_generate_polynomial_features(self):
        """Test polynomial feature generation."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        })
        
        poly_df = self.engineer.generate_polynomial_features(
            df, ['x1', 'x2'], degree=2, target_column='target'
        )
        
        # Should have original features plus polynomial combinations
        assert len(poly_df.columns) > len(df.columns)
        
        # Check for interaction terms
        interaction_cols = [col for col in poly_df.columns if 'x' in col]
        assert len(interaction_cols) > 2  # More than original features
    
    def test_generate_interaction_features(self):
        """Test interaction feature generation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [1, 1, 2, 2, 3],
            'target': [0, 1, 0, 1, 0]
        })
        
        interaction_df = self.engineer.generate_interaction_features(
            df, ['feature1', 'feature2', 'feature3'], target_column='target'
        )
        
        # Should have interaction terms
        interaction_cols = [col for col in interaction_df.columns 
                          if '_x_' in col or '*' in col]
        assert len(interaction_cols) > 0
    
    def test_generate_statistical_features(self):
        """Test statistical feature generation."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        stats_df = self.engineer.generate_statistical_features(
            df, numeric_columns=['values'], groupby_columns=['group'], target_column='target'
        )
        
        # Should have statistical features
        stats_cols = [col for col in stats_df.columns 
                     if any(stat in col for stat in ['mean', 'std', 'min', 'max'])]
        assert len(stats_cols) > 0
    
    def test_generate_time_features(self):
        """Test time-based feature generation."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.random(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        time_df = self.engineer.generate_time_features(
            df, date_columns=['date'], target_column='target'
        )
        
        # Should have time-based features
        time_cols = [col for col in time_df.columns 
                    if any(time_feat in col for time_feat in 
                          ['year', 'month', 'day', 'weekday', 'quarter'])]
        assert len(time_cols) > 0
    
    def test_generate_text_features(self):
        """Test text feature generation."""
        df = pd.DataFrame({
            'text': [
                'This is a sample text',
                'Another example text with more words',
                'Short text',
                'Very long text with many different words and characters',
                'Text with numbers 123 and symbols!'
            ],
            'target': [0, 1, 0, 1, 0]
        })
        
        text_df = self.engineer.generate_text_features(
            df, text_columns=['text'], target_column='target'
        )
        
        # Should have text-based features
        text_cols = [col for col in text_df.columns 
                    if any(text_feat in col for text_feat in 
                          ['length', 'word_count', 'char_count'])]
        assert len(text_cols) > 0
    
    def test_select_best_features(self):
        """Test feature selection based on importance."""
        # Create data with some relevant and irrelevant features
        np.random.seed(42)
        n_samples = 1000
        
        # Relevant features
        X_relevant = np.random.normal(0, 1, (n_samples, 3))
        # Irrelevant features
        X_irrelevant = np.random.normal(0, 1, (n_samples, 7))
        
        # Target correlated with relevant features
        y = (X_relevant[:, 0] + X_relevant[:, 1] > 0).astype(int)
        
        df = pd.DataFrame(
            np.column_stack([X_relevant, X_irrelevant]),
            columns=[f'feature_{i}' for i in range(10)]
        )
        df['target'] = y
        
        selected_df, feature_importance = self.engineer.select_best_features(
            df, [f'feature_{i}' for i in range(10)], 'target', k=5
        )
        
        # Should select top k features
        feature_cols = [col for col in selected_df.columns if col != 'target']
        assert len(feature_cols) == 5
        assert isinstance(feature_importance, dict)
    
    def test_automated_feature_engineering(self, mock_tabular_dataset):
        """Test automated feature engineering pipeline."""
        df, temp_file = mock_tabular_dataset
        
        # Mock dataset metadata
        metadata = {
            'data_type': DataType.TABULAR,
            'feature_types': {col: 'numeric' if col.startswith('numeric') else 'categorical' 
                            for col in df.columns if col != 'target'},
            'target_column': 'target'
        }
        
        engineered_df = self.engineer.automated_feature_engineering(
            df, metadata, max_features=20
        )
        
        assert isinstance(engineered_df, pd.DataFrame)
        # Should have more features than original (but limited by max_features)
        assert len(engineered_df.columns) <= 21  # max_features + target


class TestDataValidator:
    """Test DataValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
    
    def test_validate_dataset_format(self, mock_tabular_dataset):
        """Test dataset format validation."""
        df, temp_file = mock_tabular_dataset
        
        is_valid, errors = self.validator.validate_dataset_format(temp_file)
        
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        if not is_valid:
            assert len(errors) > 0
    
    def test_validate_data_quality(self, mock_tabular_dataset):
        """Test data quality validation."""
        df, temp_file = mock_tabular_dataset
        
        quality_report = self.validator.validate_data_quality(df)
        
        assert isinstance(quality_report, dict)
        assert 'missing_values' in quality_report
        assert 'duplicates' in quality_report
        assert 'outliers' in quality_report
        assert 'data_types' in quality_report
    
    def test_validate_feature_types(self):
        """Test feature type validation."""
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'C', 'A', 'B'],
            'mixed_col': [1, 'B', 3, 'D', 5],  # Mixed types
            'target': [0, 1, 0, 1, 0]
        })
        
        type_issues = self.validator.validate_feature_types(df)
        
        assert isinstance(type_issues, dict)
        # Should detect mixed type column
        assert 'mixed_col' in type_issues
    
    def test_validate_target_distribution(self):
        """Test target distribution validation."""
        # Balanced target
        balanced_target = [0, 1, 0, 1, 0, 1] * 100
        balance_report = self.validator.validate_target_distribution(balanced_target)
        assert balance_report['is_balanced'] is True
        
        # Imbalanced target
        imbalanced_target = [0] * 950 + [1] * 50
        imbalance_report = self.validator.validate_target_distribution(imbalanced_target)
        assert imbalance_report['is_balanced'] is False
    
    def test_detect_data_leakage(self):
        """Test data leakage detection."""
        # Create data with potential leakage
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target_copy': [0, 1, 0, 1, 0],  # Perfect correlation with target
            'target': [0, 1, 0, 1, 0]
        })
        
        leakage_features = self.validator.detect_data_leakage(
            df, ['feature1', 'feature2', 'target_copy'], 'target'
        )
        
        assert isinstance(leakage_features, list)
        # Should detect target_copy as potential leakage
        assert 'target_copy' in leakage_features
    
    def test_validate_dataset_size(self):
        """Test dataset size validation."""
        # Small dataset
        small_df = pd.DataFrame({'col1': [1, 2], 'target': [0, 1]})
        small_report = self.validator.validate_dataset_size(small_df)
        assert small_report['is_sufficient'] is False
        
        # Large enough dataset
        large_df = pd.DataFrame({
            'col1': range(1000),
            'target': [i % 2 for i in range(1000)]
        })
        large_report = self.validator.validate_dataset_size(large_df)
        assert large_report['is_sufficient'] is True


class TestDataProcessingService:
    """Test DataProcessingService integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataProcessingService()
    
    def test_process_dataset_end_to_end(self, mock_tabular_dataset):
        """Test complete dataset processing pipeline."""
        df, temp_file = mock_tabular_dataset
        
        # Process dataset
        processed_data = self.service.process_dataset(
            temp_file,
            task_type=TaskType.CLASSIFICATION,
            validation_split=0.2,
            test_split=0.1
        )
        
        assert isinstance(processed_data, dict)
        assert 'X_train' in processed_data
        assert 'X_val' in processed_data
        assert 'X_test' in processed_data
        assert 'y_train' in processed_data
        assert 'y_val' in processed_data
        assert 'y_test' in processed_data
        assert 'preprocessing_pipeline' in processed_data
        assert 'feature_names' in processed_data
        
        # Check data splits
        total_samples = (len(processed_data['X_train']) + 
                        len(processed_data['X_val']) + 
                        len(processed_data['X_test']))
        assert total_samples <= len(df)  # May be smaller due to preprocessing
    
    def test_process_dataset_with_feature_engineering(self, mock_tabular_dataset):
        """Test dataset processing with feature engineering."""
        df, temp_file = mock_tabular_dataset
        
        processed_data = self.service.process_dataset(
            temp_file,
            task_type=TaskType.CLASSIFICATION,
            enable_feature_engineering=True,
            max_features=15
        )
        
        # Should have engineered features
        assert len(processed_data['feature_names']) <= 15
    
    def test_process_dataset_regression_task(self, test_data_manager):
        """Test processing dataset for regression task."""
        # Create regression dataset
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'target': np.random.normal(0, 1, 1000)  # Continuous target
        })
        
        temp_file = test_data_manager.temp_files.append(
            tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        )
        df.to_csv(temp_file.name, index=False)
        
        processed_data = self.service.process_dataset(
            temp_file.name,
            task_type=TaskType.REGRESSION
        )
        
        assert isinstance(processed_data, dict)
        # For regression, target should be continuous
        assert processed_data['y_train'].dtype in [np.float32, np.float64]
    
    def test_get_preprocessing_recommendations(self, mock_tabular_dataset):
        """Test getting preprocessing recommendations."""
        df, temp_file = mock_tabular_dataset
        
        recommendations = self.service.get_preprocessing_recommendations(temp_file)
        
        assert isinstance(recommendations, dict)
        assert 'recommended_steps' in recommendations
        assert 'data_quality_issues' in recommendations
        assert 'estimated_processing_time' in recommendations
    
    def test_validate_processed_data(self, mock_tabular_dataset):
        """Test validation of processed data."""
        df, temp_file = mock_tabular_dataset
        
        processed_data = self.service.process_dataset(temp_file)
        
        validation_report = self.service.validate_processed_data(processed_data)
        
        assert isinstance(validation_report, dict)
        assert 'is_valid' in validation_report
        assert 'issues' in validation_report
        assert 'recommendations' in validation_report
    
    @patch('automl_framework.services.data_processing.DatasetAnalyzer')
    def test_service_with_mocked_analyzer(self, mock_analyzer_class, mock_tabular_dataset):
        """Test service with mocked analyzer."""
        df, temp_file = mock_tabular_dataset
        
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze_dataset.return_value = {
            'data_type': DataType.TABULAR,
            'n_samples': 1000,
            'n_features': 10,
            'feature_types': {'feature1': 'numeric'},
            'target_column': 'target'
        }
        mock_analyzer_class.return_value = mock_analyzer
        
        service = DataProcessingService()
        processed_data = service.process_dataset(temp_file)
        
        # Verify analyzer was called
        mock_analyzer.analyze_dataset.assert_called_once_with(temp_file)
        assert isinstance(processed_data, dict)


if __name__ == "__main__":
    pytest.main([__file__])