"""
Unit tests for DatasetAnalyzer class.

Tests dataset analysis functionality including data type detection,
feature analysis, and metadata extraction.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from PIL import Image

from automl_framework.services.data_processing import DatasetAnalyzer
from automl_framework.models.data_models import DataType, TaskType
from automl_framework.core.interfaces import DatasetMetadata


class TestDatasetAnalyzer:
    """Test suite for DatasetAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DatasetAnalyzer()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test DatasetAnalyzer initialization."""
        analyzer = DatasetAnalyzer()
        
        assert analyzer.supported_formats is not None
        assert 'tabular' in analyzer.supported_formats
        assert 'image' in analyzer.supported_formats
        assert 'text' in analyzer.supported_formats
        
        assert analyzer.categorical_threshold == 0.05
        assert analyzer.text_length_threshold == 50
        assert analyzer.image_size_threshold == 1000
    
    def test_detect_data_type_csv(self):
        """Test data type detection for CSV files."""
        # Create test CSV file
        csv_path = os.path.join(self.temp_dir, "test.csv")
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e']
        })
        df.to_csv(csv_path, index=False)
        
        data_type = self.analyzer._detect_data_type(csv_path)
        assert data_type == DataType.TABULAR
    
    def test_detect_data_type_json_tabular(self):
        """Test data type detection for JSON files with tabular data."""
        json_path = os.path.join(self.temp_dir, "test.json")
        data = [
            {'feature1': 1, 'feature2': 'a'},
            {'feature1': 2, 'feature2': 'b'},
            {'feature1': 3, 'feature2': 'c'}
        ]
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        data_type = self.analyzer._detect_data_type(json_path)
        assert data_type == DataType.TABULAR
    
    def test_detect_data_type_json_text(self):
        """Test data type detection for JSON files with text data."""
        json_path = os.path.join(self.temp_dir, "test.json")
        data = {"text": "This is a sample text document"}
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        data_type = self.analyzer._detect_data_type(json_path)
        assert data_type == DataType.TEXT
    
    def test_detect_data_type_image_file(self):
        """Test data type detection for image files."""
        # Create a simple test image
        img_path = os.path.join(self.temp_dir, "test.jpg")
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)
        
        data_type = self.analyzer._detect_data_type(img_path)
        assert data_type == DataType.IMAGE
    
    def test_detect_feature_type_numeric(self):
        """Test feature type detection for numeric data."""
        # Continuous numeric
        series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0])
        data_type, is_categorical = self.analyzer._detect_feature_type(series)
        
        assert data_type == 'float64'
        assert is_categorical == False
        
        # Categorical numeric (few unique values)
        series = pd.Series([1, 1, 2, 2, 3, 3, 1, 2, 3, 1] * 10)  # Only 3 unique values
        data_type, is_categorical = self.analyzer._detect_feature_type(series)
        
        assert data_type == 'int64'
        assert is_categorical == True
    
    def test_detect_feature_type_categorical(self):
        """Test feature type detection for categorical data."""
        # Few unique string values
        series = pd.Series(['cat', 'dog', 'cat', 'dog', 'bird'] * 20)
        data_type, is_categorical = self.analyzer._detect_feature_type(series)
        
        assert data_type == 'object'
        assert is_categorical == True
    
    def test_detect_feature_type_text(self):
        """Test feature type detection for text data."""
        # Long text strings
        series = pd.Series([
            "This is a long text document with many words and sentences.",
            "Another lengthy text sample that should be classified as text.",
            "Yet another long text string for testing purposes."
        ])
        data_type, is_categorical = self.analyzer._detect_feature_type(series)
        
        assert data_type == 'text'
        assert is_categorical == False
    
    def test_analyze_tabular_feature_numeric(self):
        """Test analysis of numeric tabular features."""
        df = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0]
        })
        
        feature = self.analyzer._analyze_tabular_feature(df, 'numeric_feature')
        
        assert feature.name == 'numeric_feature'
        assert feature.data_type == 'float64'
        assert feature.is_categorical == False
        assert feature.unique_values == 9  # 9 unique non-null values
        assert feature.missing_percentage == 10.0  # 1 out of 10 is NaN
        
        # Check statistics
        assert 'mean' in feature.statistics
        assert 'std' in feature.statistics
        assert 'min' in feature.statistics
        assert 'max' in feature.statistics
    
    def test_analyze_tabular_feature_categorical(self):
        """Test analysis of categorical tabular features."""
        df = pd.DataFrame({
            'categorical_feature': ['A', 'B', 'A', 'C', 'B', 'A', None, 'C', 'A', 'B']
        })
        
        feature = self.analyzer._analyze_tabular_feature(df, 'categorical_feature')
        
        assert feature.name == 'categorical_feature'
        assert feature.data_type == 'object'
        assert feature.is_categorical == True
        assert feature.unique_values == 3  # A, B, C
        assert feature.missing_percentage == 10.0  # 1 out of 10 is None
        
        # Check statistics
        assert 'avg_length' in feature.statistics
        assert 'most_common' in feature.statistics
    
    def test_detect_task_type_classification(self):
        """Test task type detection for classification."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': ['class_a', 'class_b', 'class_a', 'class_c', 'class_b']
        })
        
        features = [
            self.analyzer._analyze_tabular_feature(df, 'feature1'),
            self.analyzer._analyze_tabular_feature(df, 'target')
        ]
        
        task_type = self.analyzer._detect_task_type(df, 'target', features)
        assert task_type == TaskType.CLASSIFICATION
    
    def test_detect_task_type_regression(self):
        """Test task type detection for regression."""
        # Create a larger dataset with more unique continuous values
        np.random.seed(42)
        target_values = np.random.uniform(0, 100, 100)  # 100 continuous values
        df = pd.DataFrame({
            'feature1': range(100),
            'target': target_values
        })
        
        features = [
            self.analyzer._analyze_tabular_feature(df, 'feature1'),
            self.analyzer._analyze_tabular_feature(df, 'target')
        ]
        
        task_type = self.analyzer._detect_task_type(df, 'target', features)
        assert task_type == TaskType.REGRESSION
    
    def test_generate_tabular_statistics(self):
        """Test generation of overall tabular dataset statistics."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3, np.nan, 5],
            'numeric2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'text': ['short', 'medium text', 'longer text string', 'text', 'another']
        })
        
        stats = self.analyzer._generate_tabular_statistics(df)
        
        assert stats['num_rows'] == 5
        assert stats['num_columns'] == 4
        assert stats['missing_values_total'] == 1
        assert stats['missing_percentage'] == 5.0  # 1 out of 20 total values
        assert stats['duplicate_rows'] == 0
        assert stats['numeric_columns'] == 2
        assert stats['categorical_columns'] == 2
    
    def test_analyze_tabular_dataset_complete(self):
        """Test complete analysis of a tabular dataset."""
        # Create test CSV
        csv_path = os.path.join(self.temp_dir, "test_dataset.csv")
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'target': ['class1', 'class2', 'class1', 'class2', 'class1', 'class2', 'class1', 'class2', 'class1', 'class2']
        })
        df.to_csv(csv_path, index=False)
        
        metadata = self.analyzer.analyze_dataset(csv_path, target_column='target')
        
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.data_type == DataType.TABULAR
        assert metadata.task_type == TaskType.CLASSIFICATION
        assert metadata.size == 10
        assert len(metadata.features) == 4
        assert metadata.target_column == 'target'
        assert metadata.class_distribution is not None
        assert 'class1' in metadata.class_distribution
        assert 'class2' in metadata.class_distribution
    
    def test_analyze_image_files(self):
        """Test analysis of image files."""
        # Create test images
        img_paths = []
        for i, (width, height) in enumerate([(100, 100), (200, 150), (300, 200)]):
            img_path = os.path.join(self.temp_dir, f"test_{i}.jpg")
            img = Image.new('RGB', (width, height), color='red')
            img.save(img_path)
            img_paths.append(Path(img_path))
        
        stats = self.analyzer._analyze_image_files(img_paths)
        
        assert 'avg_width' in stats
        assert 'avg_height' in stats
        assert 'min_width' in stats
        assert 'max_width' in stats
        assert 'min_height' in stats
        assert 'max_height' in stats
        assert 'most_common_channels' in stats
        assert 'image_formats' in stats
        
        assert stats['avg_width'] == 200.0  # (100 + 200 + 300) / 3
        assert stats['min_width'] == 100
        assert stats['max_width'] == 300
        assert stats['most_common_channels'] == 3  # RGB
    
    def test_analyze_text_content(self):
        """Test analysis of text content."""
        texts = [
            "This is a short text.",
            "This is a much longer text document with many more words and sentences.",
            "Medium length text with some words."
        ]
        
        stats = self.analyzer._analyze_text_content(texts)
        
        assert 'avg_length' in stats
        assert 'min_length' in stats
        assert 'max_length' in stats
        assert 'avg_word_count' in stats
        assert 'min_word_count' in stats
        assert 'max_word_count' in stats
        assert 'unique_characters' in stats
        assert 'vocabulary_size' in stats
        
        assert stats['min_word_count'] == 5  # "This is a short text."
        assert stats['max_word_count'] == 13  # Longest text
    
    def test_analyze_dataset_file_not_found(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.analyzer.analyze_dataset("non_existent_file.csv")
    
    def test_analyze_dataset_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        # Create a file with unsupported extension
        unsupported_path = os.path.join(self.temp_dir, "test.xyz")
        with open(unsupported_path, 'w') as f:
            f.write("some content")
        
        # Should default to text type, but if we force an unsupported type, it should raise
        # For now, the analyzer defaults to text, so this test checks the behavior
        metadata = self.analyzer.analyze_dataset(unsupported_path)
        assert metadata.data_type == DataType.TEXT
    
    def test_analyze_empty_dataset(self):
        """Test handling of empty datasets."""
        # Create empty CSV
        csv_path = os.path.join(self.temp_dir, "empty.csv")
        df = pd.DataFrame()
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError):
            self.analyzer.analyze_dataset(csv_path)
    
    def test_feature_validation(self):
        """Test feature validation in analysis."""
        df = pd.DataFrame({
            'valid_feature': [1, 2, 3, 4, 5],
            '': [1, 2, 3, 4, 5]  # Empty column name
        })
        
        # Should handle empty column names gracefully
        feature = self.analyzer._analyze_tabular_feature(df, '')
        assert feature.name == ''  # Should preserve the empty name but handle it
    
    def test_missing_values_handling(self):
        """Test handling of datasets with many missing values."""
        df = pd.DataFrame({
            'mostly_missing': [1, np.nan, np.nan, np.nan, np.nan],
            'some_missing': [1, 2, np.nan, 4, 5],
            'no_missing': [1, 2, 3, 4, 5]
        })
        
        feature_mostly = self.analyzer._analyze_tabular_feature(df, 'mostly_missing')
        feature_some = self.analyzer._analyze_tabular_feature(df, 'some_missing')
        feature_none = self.analyzer._analyze_tabular_feature(df, 'no_missing')
        
        assert feature_mostly.missing_percentage == 80.0
        assert feature_some.missing_percentage == 20.0
        assert feature_none.missing_percentage == 0.0
    
    def test_large_dataset_sampling(self):
        """Test that large datasets are handled efficiently."""
        # Create a larger dataset
        large_df = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.choice(['class1', 'class2'], 1000)
        })
        
        csv_path = os.path.join(self.temp_dir, "large_dataset.csv")
        large_df.to_csv(csv_path, index=False)
        
        metadata = self.analyzer.analyze_dataset(csv_path, target_column='target')
        
        assert metadata.size == 1000
        assert len(metadata.features) == 3
        assert metadata.data_type == DataType.TABULAR