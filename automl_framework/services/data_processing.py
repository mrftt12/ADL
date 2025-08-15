"""
Data processing service for automated dataset analysis and preprocessing.

This module provides the DatasetAnalyzer class for analyzing dataset characteristics,
detecting data types, and extracting metadata for the AutoML pipeline.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from PIL import Image
import json
import re

from ..models.data_models import Dataset, Feature, DataType, TaskType
from ..core.interfaces import DatasetMetadata, IDataProcessor, ProcessedData


logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Analyzes datasets to extract metadata, detect data types, and provide
    statistical information for automated preprocessing.
    """
    
    def __init__(self):
        """Initialize the DatasetAnalyzer."""
        self.supported_formats = {
            'tabular': ['.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.json'],
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'],
            'text': ['.txt', '.json', '.jsonl']
        }
        
        # Thresholds for feature type detection
        self.categorical_threshold = 0.05  # If unique values / total < threshold, likely categorical
        self.text_length_threshold = 50    # If avg string length > threshold, likely text
        self.image_size_threshold = 1000   # Minimum file size for images (bytes)
    
    def analyze_dataset(self, dataset_path: str, target_column: Optional[str] = None) -> DatasetMetadata:
        """
        Analyze a dataset and return comprehensive metadata.
        
        Args:
            dataset_path: Path to the dataset file
            target_column: Optional target column name for supervised learning
            
        Returns:
            DatasetMetadata object containing analysis results
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset format is not supported
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Detect data type based on file extension and content
        data_type = self._detect_data_type(dataset_path)
        
        if data_type == DataType.TABULAR:
            return self._analyze_tabular_dataset(dataset_path, target_column)
        elif data_type == DataType.IMAGE:
            return self._analyze_image_dataset(dataset_path, target_column)
        elif data_type == DataType.TEXT:
            return self._analyze_text_dataset(dataset_path, target_column)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _detect_data_type(self, dataset_path: str) -> DataType:
        """
        Detect the data type based on file extension and content analysis.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            DataType enum value
        """
        file_ext = Path(dataset_path).suffix.lower()
        
        # Check if it's a directory (likely image dataset)
        if os.path.isdir(dataset_path):
            # Check if directory contains images
            image_files = []
            for ext in self.supported_formats['image']:
                image_files.extend(Path(dataset_path).glob(f"**/*{ext}"))
            
            if image_files:
                return DataType.IMAGE
        
        # Check file extension
        if file_ext in self.supported_formats['tabular']:
            # For JSON files, need to check content to distinguish tabular vs text
            if file_ext == '.json':
                try:
                    with open(dataset_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                            return DataType.TABULAR
                        else:
                            return DataType.TEXT
                except Exception:
                    return DataType.TEXT
            else:
                return DataType.TABULAR
        elif file_ext in self.supported_formats['image']:
            return DataType.IMAGE
        elif file_ext in self.supported_formats['text']:
            return DataType.TEXT
        
        # If extension is ambiguous, analyze content
        try:
            # Try to read as tabular data
            if file_ext in ['.csv', '.tsv']:
                pd.read_csv(dataset_path, nrows=5)
                return DataType.TABULAR
        except Exception:
            pass
        
        # Default to text if can't determine
        return DataType.TEXT
    
    def _analyze_tabular_dataset(self, dataset_path: str, target_column: Optional[str] = None) -> DatasetMetadata:
        """
        Analyze tabular dataset (CSV, Excel, etc.).
        
        Args:
            dataset_path: Path to the tabular dataset
            target_column: Optional target column name
            
        Returns:
            DatasetMetadata for tabular dataset
        """
        # Load dataset
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.tsv'):
                df = pd.read_csv(dataset_path, sep='\t')
            elif dataset_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(dataset_path)
            elif dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            else:
                raise ValueError(f"Unsupported tabular format: {dataset_path}")
        except Exception as e:
            raise ValueError(f"Failed to load tabular dataset: {e}")
        
        # Basic dataset info
        dataset_size = len(df)
        dataset_name = Path(dataset_path).stem
        
        # Analyze features
        features = []
        for column in df.columns:
            feature = self._analyze_tabular_feature(df, column)
            features.append(feature)
        
        # Detect task type
        task_type = self._detect_task_type(df, target_column, features)
        
        # Calculate class distribution if classification task
        class_distribution = None
        if task_type in [TaskType.CLASSIFICATION] and target_column:
            class_distribution = df[target_column].value_counts().to_dict()
        
        # Generate overall statistics
        statistics = self._generate_tabular_statistics(df)
        
        return DatasetMetadata(
            id=dataset_name,
            name=dataset_name,
            data_type=DataType.TABULAR,
            task_type=task_type,
            size=dataset_size,
            features=features,
            target_column=target_column,
            class_distribution=class_distribution,
            statistics=statistics
        )
    
    def _analyze_tabular_feature(self, df: pd.DataFrame, column: str) -> Feature:
        """
        Analyze a single feature in a tabular dataset.
        
        Args:
            df: DataFrame containing the data
            column: Column name to analyze
            
        Returns:
            Feature object with analysis results
        """
        series = df[column]
        
        # Basic statistics
        missing_values = series.isnull().sum()
        unique_values = series.nunique()
        total_values = len(series)
        
        # Detect data type and if categorical
        data_type, is_categorical = self._detect_feature_type(series)
        
        # Generate feature-specific statistics
        statistics = {}
        
        if data_type in ['int64', 'float64']:
            # Numerical statistics
            statistics.update({
                'mean': float(series.mean()) if not series.empty else 0.0,
                'std': float(series.std()) if not series.empty else 0.0,
                'min': float(series.min()) if not series.empty else 0.0,
                'max': float(series.max()) if not series.empty else 0.0,
                'median': float(series.median()) if not series.empty else 0.0,
                'q25': float(series.quantile(0.25)) if not series.empty else 0.0,
                'q75': float(series.quantile(0.75)) if not series.empty else 0.0,
                'skewness': float(series.skew()) if not series.empty else 0.0,
                'kurtosis': float(series.kurtosis()) if not series.empty else 0.0
            })
        elif data_type == 'object':
            # String/categorical statistics
            if not series.empty:
                str_lengths = series.astype(str).str.len()
                statistics.update({
                    'avg_length': float(str_lengths.mean()),
                    'min_length': int(str_lengths.min()),
                    'max_length': int(str_lengths.max()),
                    'most_common': series.value_counts().head(5).to_dict()
                })
        
        return Feature(
            name=column,
            data_type=data_type,
            is_categorical=is_categorical,
            unique_values=unique_values,
            missing_percentage=float(missing_values / total_values * 100),
            statistics=statistics
        )
    
    def _detect_feature_type(self, series: pd.Series) -> Tuple[str, bool]:
        """
        Detect the data type and whether a feature is categorical.
        
        Args:
            series: Pandas series to analyze
            
        Returns:
            Tuple of (data_type, is_categorical)
        """
        # Get pandas dtype
        dtype = str(series.dtype)
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's actually categorical (few unique values)
            unique_ratio = series.nunique() / len(series)
            is_categorical = unique_ratio < self.categorical_threshold
            return dtype, is_categorical
        
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime64', False
        
        # For object type, determine if text or categorical
        if dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            
            # If few unique values, likely categorical
            if unique_ratio < self.categorical_threshold:
                return 'object', True
            
            # Check average string length to distinguish text from categorical
            if not series.empty:
                avg_length = series.astype(str).str.len().mean()
                if avg_length > self.text_length_threshold:
                    return 'text', False
                else:
                    return 'object', True
            
            return 'object', False
        
        return dtype, False
    
    def _detect_task_type(self, df: pd.DataFrame, target_column: Optional[str], features: List[Feature]) -> TaskType:
        """
        Detect the machine learning task type based on target column analysis.
        
        Args:
            df: DataFrame containing the data
            target_column: Target column name
            features: List of analyzed features
            
        Returns:
            TaskType enum value
        """
        if not target_column or target_column not in df.columns:
            # No target column specified, assume classification for now
            return TaskType.CLASSIFICATION
        
        target_series = df[target_column]
        
        # Find the target feature
        target_feature = None
        for feature in features:
            if feature.name == target_column:
                target_feature = feature
                break
        
        if not target_feature:
            return TaskType.CLASSIFICATION
        
        # Determine task type based on target characteristics
        if target_feature.data_type in ['int64', 'float64']:
            # Numeric target - check if it's truly continuous or discrete
            unique_ratio = target_feature.unique_values / len(df)
            if target_feature.is_categorical or target_feature.unique_values <= 20 or unique_ratio < 0.1:
                return TaskType.CLASSIFICATION
            else:
                return TaskType.REGRESSION
        else:
            # Non-numeric target, likely classification
            return TaskType.CLASSIFICATION
    
    def _generate_tabular_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate overall dataset statistics for tabular data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of dataset statistics
        """
        return {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'missing_values_total': int(df.isnull().sum().sum()),
            'missing_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'duplicate_rows': int(df.duplicated().sum()),
            'numeric_columns': int(df.select_dtypes(include=[np.number]).shape[1]),
            'categorical_columns': int(df.select_dtypes(include=['object']).shape[1]),
            'datetime_columns': int(df.select_dtypes(include=['datetime64']).shape[1])
        }
    
    def _analyze_image_dataset(self, dataset_path: str, target_column: Optional[str] = None) -> DatasetMetadata:
        """
        Analyze image dataset (directory of images or single image).
        
        Args:
            dataset_path: Path to image dataset
            target_column: Optional target column (for classification)
            
        Returns:
            DatasetMetadata for image dataset
        """
        dataset_name = Path(dataset_path).stem
        
        if os.path.isdir(dataset_path):
            # Directory of images
            image_files = []
            for ext in self.supported_formats['image']:
                image_files.extend(Path(dataset_path).glob(f"**/*{ext}"))
            
            if not image_files:
                raise ValueError(f"No image files found in directory: {dataset_path}")
            
            # Analyze sample of images for characteristics
            sample_size = min(100, len(image_files))
            sample_files = np.random.choice(image_files, sample_size, replace=False)
            
            image_stats = self._analyze_image_files(sample_files)
            
            # Detect if it's a classification task based on directory structure
            task_type = TaskType.CLASSIFICATION
            class_distribution = None
            
            # Check if images are organized in class directories
            class_dirs = [d for d in Path(dataset_path).iterdir() if d.is_dir()]
            if class_dirs:
                class_distribution = {}
                for class_dir in class_dirs:
                    class_images = []
                    for ext in self.supported_formats['image']:
                        class_images.extend(class_dir.glob(f"*{ext}"))
                    class_distribution[class_dir.name] = len(class_images)
            
            features = [
                Feature(
                    name="image_data",
                    data_type="image",
                    is_categorical=False,
                    unique_values=len(image_files),
                    missing_percentage=0.0,
                    statistics=image_stats
                )
            ]
            
            return DatasetMetadata(
                id=dataset_name,
                name=dataset_name,
                data_type=DataType.IMAGE,
                task_type=task_type,
                size=len(image_files),
                features=features,
                target_column=target_column,
                class_distribution=class_distribution,
                statistics={
                    'total_images': len(image_files),
                    'total_size_mb': sum(f.stat().st_size for f in image_files) / 1024 / 1024,
                    **image_stats
                }
            )
        else:
            # Single image file
            image_stats = self._analyze_image_files([Path(dataset_path)])
            
            features = [
                Feature(
                    name="image_data",
                    data_type="image",
                    is_categorical=False,
                    unique_values=1,
                    missing_percentage=0.0,
                    statistics=image_stats
                )
            ]
            
            return DatasetMetadata(
                id=dataset_name,
                name=dataset_name,
                data_type=DataType.IMAGE,
                task_type=TaskType.CLASSIFICATION,
                size=1,
                features=features,
                target_column=target_column,
                class_distribution=None,
                statistics={
                    'total_images': 1,
                    'total_size_mb': Path(dataset_path).stat().st_size / 1024 / 1024,
                    **image_stats
                }
            )
    
    def _analyze_image_files(self, image_files: List[Path]) -> Dict[str, Any]:
        """
        Analyze a collection of image files to extract characteristics.
        
        Args:
            image_files: List of image file paths
            
        Returns:
            Dictionary of image statistics
        """
        widths, heights, channels, formats = [], [], [], []
        
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
                    
                    # Detect number of channels
                    if img.mode == 'RGB':
                        channels.append(3)
                    elif img.mode == 'RGBA':
                        channels.append(4)
                    elif img.mode in ['L', 'P']:
                        channels.append(1)
                    else:
                        channels.append(3)  # Default
                    
                    formats.append(img.format.lower() if img.format else 'unknown')
            except Exception as e:
                logger.warning(f"Failed to analyze image {img_path}: {e}")
                continue
        
        if not widths:
            return {'error': 'No valid images found'}
        
        return {
            'avg_width': float(np.mean(widths)),
            'avg_height': float(np.mean(heights)),
            'min_width': int(np.min(widths)),
            'max_width': int(np.max(widths)),
            'min_height': int(np.min(heights)),
            'max_height': int(np.max(heights)),
            'most_common_channels': int(max(set(channels), key=channels.count)),
            'image_formats': list(set(formats)),
            'aspect_ratios': [w/h for w, h in zip(widths, heights)]
        }
    
    def _analyze_text_dataset(self, dataset_path: str, target_column: Optional[str] = None) -> DatasetMetadata:
        """
        Analyze text dataset (text files, JSON with text data).
        
        Args:
            dataset_path: Path to text dataset
            target_column: Optional target column
            
        Returns:
            DatasetMetadata for text dataset
        """
        dataset_name = Path(dataset_path).stem
        
        # Load text data
        texts = []
        labels = []
        
        try:
            if dataset_path.endswith('.txt'):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    texts = f.readlines()
            elif dataset_path.endswith('.xyz'):
                # Handle unknown extensions as plain text
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    texts = f.readlines()
            elif dataset_path.endswith('.json'):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                # Extract text and labels from JSON structure
                                text_fields = ['text', 'content', 'message', 'document']
                                label_fields = ['label', 'class', 'category', 'sentiment']
                                
                                text = None
                                for field in text_fields:
                                    if field in item:
                                        text = item[field]
                                        break
                                
                                if text:
                                    texts.append(text)
                                    
                                    # Extract label if available
                                    label = None
                                    for field in label_fields:
                                        if field in item:
                                            label = item[field]
                                            break
                                    labels.append(label)
                            else:
                                texts.append(str(item))
                    else:
                        texts = [str(data)]
            elif dataset_path.endswith('.jsonl'):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        if isinstance(item, dict):
                            # Similar extraction logic as JSON
                            text_fields = ['text', 'content', 'message', 'document']
                            for field in text_fields:
                                if field in item:
                                    texts.append(item[field])
                                    break
                        else:
                            texts.append(str(item))
        except Exception as e:
            raise ValueError(f"Failed to load text dataset: {e}")
        
        if not texts:
            raise ValueError("No text data found in dataset")
        
        # Analyze text characteristics
        text_stats = self._analyze_text_content(texts)
        
        # Determine task type
        task_type = TaskType.CLASSIFICATION  # Default for text
        if any(labels):
            task_type = TaskType.CLASSIFICATION
        
        # Calculate class distribution if labels available
        class_distribution = None
        if labels and any(l is not None for l in labels):
            valid_labels = [l for l in labels if l is not None]
            class_distribution = pd.Series(valid_labels).value_counts().to_dict()
        
        features = [
            Feature(
                name="text_content",
                data_type="text",
                is_categorical=False,
                unique_values=len(set(texts)),
                missing_percentage=0.0,
                statistics=text_stats
            )
        ]
        
        if labels and any(l is not None for l in labels):
            label_stats = pd.Series([l for l in labels if l is not None])
            features.append(
                Feature(
                    name="label",
                    data_type="object",
                    is_categorical=True,
                    unique_values=label_stats.nunique(),
                    missing_percentage=float(labels.count(None) / len(labels) * 100),
                    statistics={'most_common': label_stats.value_counts().head(5).to_dict()}
                )
            )
        
        return DatasetMetadata(
            id=dataset_name,
            name=dataset_name,
            data_type=DataType.TEXT,
            task_type=task_type,
            size=len(texts),
            features=features,
            target_column=target_column or 'label' if labels else None,
            class_distribution=class_distribution,
            statistics={
                'total_documents': len(texts),
                'total_size_mb': sum(len(t.encode('utf-8')) for t in texts) / 1024 / 1024,
                **text_stats
            }
        )
    
    def _analyze_text_content(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze text content to extract linguistic characteristics.
        
        Args:
            texts: List of text documents
            
        Returns:
            Dictionary of text statistics
        """
        if not texts:
            return {}
        
        # Basic text statistics
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # Character and word analysis
        all_text = ' '.join(texts)
        unique_chars = len(set(all_text))
        
        # Language detection (basic)
        ascii_ratio = sum(1 for c in all_text if ord(c) < 128) / len(all_text) if all_text else 0
        
        return {
            'avg_length': float(np.mean(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'std_length': float(np.std(lengths)),
            'avg_word_count': float(np.mean(word_counts)),
            'min_word_count': int(np.min(word_counts)),
            'max_word_count': int(np.max(word_counts)),
            'unique_characters': unique_chars,
            'ascii_ratio': float(ascii_ratio),
            'total_words': sum(word_counts),
            'vocabulary_size': len(set(all_text.split()))
        }

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import joblib


class PreprocessingTransformer:
    """
    Base class for preprocessing transformers.
    """
    
    def __init__(self, name: str):
        """Initialize transformer with a name."""
        self.name = name
        self.fitted = False
    
    def fit(self, data: pd.DataFrame) -> 'PreprocessingTransformer':
        """Fit the transformer to the data."""
        raise NotImplementedError("Subclasses must implement fit method")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        raise NotImplementedError("Subclasses must implement transform method")
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(data).transform(data)
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {'name': self.name}


class MissingValueImputer(PreprocessingTransformer):
    """
    Transformer for handling missing values.
    """
    
    def __init__(self, strategy: str = 'auto', fill_value=None):
        """
        Initialize missing value imputer.
        
        Args:
            strategy: Imputation strategy ('mean', 'median', 'mode', 'constant', 'knn', 'auto')
            fill_value: Value to use for constant strategy
        """
        super().__init__('missing_value_imputer')
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputers = {}
    
    def fit(self, data: pd.DataFrame) -> 'MissingValueImputer':
        """Fit imputers for each column."""
        for column in data.columns:
            if data[column].isnull().any() or (data[column] == None).any():
                strategy = self._determine_strategy(data[column])
                
                # Check if all values are missing
                if data[column].isnull().all():
                    # For all-missing columns, use constant strategy
                    if pd.api.types.is_numeric_dtype(data[column]):
                        imputer = SimpleImputer(strategy='constant', fill_value=0)
                    else:
                        imputer = SimpleImputer(strategy='constant', fill_value='missing')
                else:
                    if strategy in ['mean', 'median']:
                        imputer = SimpleImputer(strategy=strategy)
                    elif strategy == 'most_frequent':
                        imputer = SimpleImputer(strategy='most_frequent')
                    elif strategy == 'constant':
                        imputer = SimpleImputer(strategy='constant', fill_value=self.fill_value)
                    elif strategy == 'knn':
                        imputer = KNNImputer(n_neighbors=5)
                    else:
                        # Default to median for numeric, mode for categorical
                        if pd.api.types.is_numeric_dtype(data[column]):
                            imputer = SimpleImputer(strategy='median')
                        else:
                            imputer = SimpleImputer(strategy='most_frequent')
                
                # Fit imputer
                try:
                    if strategy == 'knn' and not data[column].isnull().all():
                        # KNN imputer needs numeric data
                        if pd.api.types.is_numeric_dtype(data[column]):
                            imputer.fit(data[[column]])
                        else:
                            # Use mode for non-numeric with KNN
                            imputer = SimpleImputer(strategy='most_frequent')
                            imputer.fit(data[[column]])
                    else:
                        # For categorical data, ensure we handle None properly
                        if data[column].dtype == 'object':
                            # Replace None with np.nan for proper handling
                            column_data = data[[column]].replace({None: np.nan})
                            imputer.fit(column_data)
                        else:
                            imputer.fit(data[[column]])
                except ValueError:
                    # If fitting fails (e.g., all missing), use constant imputer
                    if pd.api.types.is_numeric_dtype(data[column]):
                        imputer = SimpleImputer(strategy='constant', fill_value=0)
                    else:
                        imputer = SimpleImputer(strategy='constant', fill_value='missing')
                    imputer.fit(data[[column]])
                
                self.imputers[column] = imputer
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values."""
        if not self.fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        result = data.copy()
        for column, imputer in self.imputers.items():
            if column in result.columns and (result[column].isnull().any() or (result[column] == None).any()):
                # Handle categorical data properly
                if result[column].dtype == 'object':
                    # For object columns, replace None with np.nan first
                    column_data = result[[column]].replace({None: np.nan})
                    try:
                        transformed = imputer.transform(column_data)
                        if transformed.size > 0:
                            result[column] = transformed.flatten()
                    except ValueError:
                        # If all values are missing, fill with a default value
                        if result[column].isnull().all():
                            result[column] = result[column].fillna('missing')
                else:
                    try:
                        transformed = imputer.transform(result[[column]])
                        if transformed.size > 0:
                            result[column] = transformed.flatten()
                    except ValueError:
                        # If all values are missing, fill with 0
                        if result[column].isnull().all():
                            result[column] = result[column].fillna(0)
        
        return result
    
    def _determine_strategy(self, series: pd.Series) -> str:
        """Determine the best imputation strategy for a series."""
        if self.strategy != 'auto':
            return self.strategy
        
        # Auto strategy selection
        if pd.api.types.is_numeric_dtype(series):
            # For numeric data, use median (robust to outliers)
            return 'median'
        else:
            # For categorical data, use most frequent
            return 'most_frequent'


class NumericalScaler(PreprocessingTransformer):
    """
    Transformer for scaling numerical features.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize numerical scaler.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust', 'auto')
        """
        super().__init__('numerical_scaler')
        self.method = method
        self.scalers = {}
        self.numeric_columns = []
    
    def fit(self, data: pd.DataFrame) -> 'NumericalScaler':
        """Fit scalers for numerical columns."""
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in self.numeric_columns:
            method = self._determine_scaling_method(data[column])
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()  # Default
            
            scaler.fit(data[[column]])
            self.scalers[column] = scaler
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by scaling numerical features."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        result = data.copy()
        for column, scaler in self.scalers.items():
            if column in result.columns:
                result[column] = scaler.transform(result[[column]]).flatten()
        
        return result
    
    def _determine_scaling_method(self, series: pd.Series) -> str:
        """Determine the best scaling method for a series."""
        if self.method != 'auto':
            return self.method
        
        # Auto method selection based on data distribution
        if len(series.dropna()) < 10:
            return 'standard'  # Default for small samples
        
        # Check for outliers using IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        outliers = ((series < (Q1 - outlier_threshold)) | (series > (Q3 + outlier_threshold))).sum()
        outlier_ratio = outliers / len(series)
        
        if outlier_ratio > 0.1:  # More than 10% outliers
            return 'robust'
        elif series.min() >= 0 and series.max() <= 1:  # Already in [0,1] range
            return 'standard'  # Don't change much
        else:
            return 'standard'  # Default


class CategoricalEncoder(PreprocessingTransformer):
    """
    Transformer for encoding categorical features.
    """
    
    def __init__(self, method: str = 'auto', max_categories: int = 10):
        """
        Initialize categorical encoder.
        
        Args:
            method: Encoding method ('onehot', 'label', 'auto')
            max_categories: Maximum categories for one-hot encoding
        """
        super().__init__('categorical_encoder')
        self.method = method
        self.max_categories = max_categories
        self.encoders = {}
        self.categorical_columns = []
        self.encoding_methods = {}
    
    def fit(self, data: pd.DataFrame) -> 'CategoricalEncoder':
        """Fit encoders for categorical columns."""
        # Identify categorical columns
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for column in self.categorical_columns:
            method = self._determine_encoding_method(data[column])
            self.encoding_methods[column] = method
            
            if method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(data[[column]])
            elif method == 'label':
                encoder = LabelEncoder()
                encoder.fit(data[column].fillna('missing'))
            else:
                encoder = LabelEncoder()  # Default
                encoder.fit(data[column].fillna('missing'))
            
            self.encoders[column] = encoder
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by encoding categorical features."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        result = data.copy()
        
        for column, encoder in self.encoders.items():
            if column not in result.columns:
                continue
            
            method = self.encoding_methods[column]
            
            if method == 'onehot':
                # One-hot encoding
                encoded = encoder.transform(result[[column]])
                feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=result.index)
                
                # Drop original column and add encoded columns
                result = result.drop(columns=[column])
                result = pd.concat([result, encoded_df], axis=1)
            
            elif method == 'label':
                # Label encoding
                result[column] = encoder.transform(result[column].fillna('missing'))
        
        return result
    
    def _determine_encoding_method(self, series: pd.Series) -> str:
        """Determine the best encoding method for a series."""
        if self.method != 'auto':
            return self.method
        
        # Auto method selection
        unique_count = series.nunique()
        
        if unique_count <= self.max_categories:
            return 'onehot'
        else:
            return 'label'


class PreprocessingPipeline:
    """
    Main preprocessing pipeline that orchestrates multiple transformers.
    """
    
    def __init__(self, transformers: Optional[List[PreprocessingTransformer]] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            transformers: List of preprocessing transformers to apply
        """
        self.transformers = transformers or []
        self.fitted = False
        self.feature_names_in = []
        self.feature_names_out = []
    
    def add_transformer(self, transformer: PreprocessingTransformer):
        """Add a transformer to the pipeline."""
        self.transformers.append(transformer)
    
    def fit(self, data: pd.DataFrame) -> 'PreprocessingPipeline':
        """Fit all transformers in the pipeline."""
        self.feature_names_in = data.columns.tolist()
        current_data = data.copy()
        
        for transformer in self.transformers:
            transformer.fit(current_data)
            current_data = transformer.transform(current_data)
        
        self.feature_names_out = current_data.columns.tolist()
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data through all transformers."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        result = data.copy()
        for transformer in self.transformers:
            result = transformer.transform(result)
        
        return result
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(data).transform(data)
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        return self.feature_names_out
    
    def save(self, filepath: str):
        """Save the fitted pipeline to disk."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'PreprocessingPipeline':
        """Load a fitted pipeline from disk."""
        return joblib.load(filepath)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline to dictionary."""
        return {
            'transformers': [t.get_params() for t in self.transformers],
            'fitted': self.fitted,
            'feature_names_in': self.feature_names_in,
            'feature_names_out': self.feature_names_out
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PreprocessingPipeline':
        """Deserialize pipeline from dictionary."""
        # This is a simplified version - in practice, you'd need to reconstruct
        # the actual transformer objects with their fitted state
        pipeline = cls()
        pipeline.fitted = config.get('fitted', False)
        pipeline.feature_names_in = config.get('feature_names_in', [])
        pipeline.feature_names_out = config.get('feature_names_out', [])
        return pipeline


class AutoPreprocessor:
    """
    Automated preprocessing that selects appropriate transformers based on data characteristics.
    """
    
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.2, random_state: int = 42):
        """
        Initialize auto preprocessor.
        
        Args:
            test_size: Proportion of data for test set
            validation_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.pipeline = None
    
    def create_preprocessing_pipeline(self, metadata: DatasetMetadata) -> PreprocessingPipeline:
        """
        Create preprocessing pipeline based on dataset metadata.
        
        Args:
            metadata: Dataset metadata from analysis
            
        Returns:
            Configured preprocessing pipeline
        """
        pipeline = PreprocessingPipeline()
        
        # Determine which transformers to add based on data characteristics
        needs_imputation = any(f.missing_percentage > 0 for f in metadata.features)
        has_numerical = any(f.data_type in ['int64', 'float64'] and not f.is_categorical 
                           for f in metadata.features)
        has_categorical = any(f.is_categorical or f.data_type == 'object' 
                             for f in metadata.features)
        
        # Add missing value imputation if needed
        if needs_imputation:
            imputer = MissingValueImputer(strategy='auto')
            pipeline.add_transformer(imputer)
        
        # Add categorical encoding if needed
        if has_categorical:
            encoder = CategoricalEncoder(method='auto', max_categories=10)
            pipeline.add_transformer(encoder)
        
        # Add numerical scaling if needed
        if has_numerical:
            scaler = NumericalScaler(method='auto')
            pipeline.add_transformer(scaler)
        
        self.pipeline = pipeline
        return pipeline
    
    def apply_preprocessing(self, pipeline: PreprocessingPipeline, data: pd.DataFrame, 
                          target_column: Optional[str] = None) -> ProcessedData:
        """
        Apply preprocessing pipeline to data and split into train/validation/test sets.
        
        Args:
            pipeline: Fitted preprocessing pipeline
            data: Input data
            target_column: Target column name
            
        Returns:
            ProcessedData object with split datasets
        """
        # Separate features and target
        if target_column and target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            X = data
            y = None
        
        # Apply preprocessing to features
        X_processed = pipeline.fit_transform(X)
        
        # Split data
        if y is not None:
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=self.test_size, random_state=self.random_state,
                stratify=y if len(y.unique()) < 20 else None  # Stratify for classification
            )
            
            # Split train into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.validation_size, random_state=self.random_state,
                stratify=y_train if len(y_train.unique()) < 20 else None
            )
            
            # Create DataFrames with target
            # Reset indices to ensure proper concatenation
            X_train = X_train.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_val = y_val.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
            
            # Add target column back
            train_data = X_train.copy()
            train_data[target_column] = y_train
            
            val_data = X_val.copy()
            val_data[target_column] = y_val
            
            test_data = X_test.copy()
            test_data[target_column] = y_test
        else:
            # No target column, just split features
            X_train, X_test = train_test_split(
                X_processed, test_size=self.test_size, random_state=self.random_state
            )
            
            X_train, X_val = train_test_split(
                X_train, test_size=self.validation_size, random_state=self.random_state
            )
            
            train_data = X_train
            val_data = X_val
            test_data = X_test
        
        return ProcessedData(
            train_data=train_data,
            validation_data=val_data,
            test_data=test_data,
            preprocessing_pipeline=pipeline,
            feature_names=pipeline.get_feature_names_out()
        )


# Update DatasetAnalyzer to implement IDataProcessor interface
class DataProcessingService(IDataProcessor):
    """
    Complete data processing service that combines analysis and preprocessing.
    """
    
    def __init__(self):
        """Initialize the data processing service."""
        self.analyzer = DatasetAnalyzer()
        self.preprocessor = AutoPreprocessor()
    
    def analyze_dataset(self, dataset_path: str) -> DatasetMetadata:
        """Analyze dataset and return metadata."""
        return self.analyzer.analyze_dataset(dataset_path)
    
    def create_preprocessing_pipeline(self, metadata: DatasetMetadata) -> PreprocessingPipeline:
        """Create preprocessing pipeline based on metadata."""
        return self.preprocessor.create_preprocessing_pipeline(metadata)
    
    def apply_preprocessing(self, pipeline: PreprocessingPipeline, data: pd.DataFrame) -> ProcessedData:
        """Apply preprocessing pipeline to data."""
        return self.preprocessor.apply_preprocessing(pipeline, data)
    
    def process_dataset(self, dataset_path: str, target_column: Optional[str] = None) -> Tuple[DatasetMetadata, ProcessedData]:
        """
        Complete dataset processing workflow.
        
        Args:
            dataset_path: Path to dataset file
            target_column: Optional target column name
            
        Returns:
            Tuple of (metadata, processed_data)
        """
        # Step 1: Analyze dataset
        metadata = self.analyzer.analyze_dataset(dataset_path, target_column)
        
        # Step 2: Load data
        if metadata.data_type == DataType.TABULAR:
            if dataset_path.endswith('.csv'):
                data = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                data = pd.read_json(dataset_path)
            else:
                raise ValueError(f"Unsupported tabular format: {dataset_path}")
        else:
            raise ValueError(f"Processing not yet implemented for {metadata.data_type}")
        
        # Step 3: Create and apply preprocessing pipeline
        pipeline = self.create_preprocessing_pipeline(metadata)
        processed_data = self.preprocessor.apply_preprocessing(pipeline, data, target_column)
        
        return metadata, processed_data


from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
import itertools
from scipy import stats


class FeatureGenerator:
    """
    Base class for feature generators.
    """
    
    def __init__(self, name: str):
        """Initialize feature generator with a name."""
        self.name = name
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate new features from the input data."""
        raise NotImplementedError("Subclasses must implement generate_features method")
    
    def get_feature_names(self, input_features: List[str]) -> List[str]:
        """Get names of generated features."""
        raise NotImplementedError("Subclasses must implement get_feature_names method")


class NumericalFeatureGenerator(FeatureGenerator):
    """
    Generator for numerical features including polynomial, interaction, and statistical features.
    """
    
    def __init__(self, include_polynomial: bool = True, include_interactions: bool = True,
                 include_statistical: bool = True, polynomial_degree: int = 2,
                 max_interactions: int = 10):
        """
        Initialize numerical feature generator.
        
        Args:
            include_polynomial: Whether to include polynomial features
            include_interactions: Whether to include interaction features
            include_statistical: Whether to include statistical features
            polynomial_degree: Degree for polynomial features
            max_interactions: Maximum number of interaction features to create
        """
        super().__init__('numerical_feature_generator')
        self.include_polynomial = include_polynomial
        self.include_interactions = include_interactions
        self.include_statistical = include_statistical
        self.polynomial_degree = polynomial_degree
        self.max_interactions = max_interactions
        self.generated_feature_names = []
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate numerical features."""
        result = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return result
        
        self.generated_feature_names = []
        
        # Polynomial features
        if self.include_polynomial and len(numeric_columns) > 0:
            result = self._add_polynomial_features(result, numeric_columns)
        
        # Interaction features
        if self.include_interactions and len(numeric_columns) > 1:
            result = self._add_interaction_features(result, numeric_columns)
        
        # Statistical features
        if self.include_statistical and len(numeric_columns) > 1:
            result = self._add_statistical_features(result, numeric_columns)
        
        return result
    
    def _add_polynomial_features(self, data: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Add polynomial features."""
        result = data.copy()
        
        for column in numeric_columns:
            for degree in range(2, self.polynomial_degree + 1):
                feature_name = f"{column}_poly_{degree}"
                result[feature_name] = data[column] ** degree
                self.generated_feature_names.append(feature_name)
        
        return result
    
    def _add_interaction_features(self, data: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Add interaction features."""
        result = data.copy()
        interactions_added = 0
        
        # Generate pairwise interactions
        for col1, col2 in itertools.combinations(numeric_columns, 2):
            if interactions_added >= self.max_interactions:
                break
            
            # Multiplication interaction
            feature_name = f"{col1}_x_{col2}"
            result[feature_name] = data[col1] * data[col2]
            self.generated_feature_names.append(feature_name)
            interactions_added += 1
            
            if interactions_added >= self.max_interactions:
                break
            
            # Division interaction (avoid division by zero)
            feature_name = f"{col1}_div_{col2}"
            result[feature_name] = data[col1] / (data[col2] + 1e-8)
            self.generated_feature_names.append(feature_name)
            interactions_added += 1
        
        return result
    
    def _add_statistical_features(self, data: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Add statistical features across numeric columns."""
        result = data.copy()
        
        # Row-wise statistics
        numeric_data = data[numeric_columns]
        
        result['row_mean'] = numeric_data.mean(axis=1)
        result['row_std'] = numeric_data.std(axis=1)
        result['row_min'] = numeric_data.min(axis=1)
        result['row_max'] = numeric_data.max(axis=1)
        result['row_median'] = numeric_data.median(axis=1)
        result['row_range'] = result['row_max'] - result['row_min']
        
        statistical_features = ['row_mean', 'row_std', 'row_min', 'row_max', 'row_median', 'row_range']
        self.generated_feature_names.extend(statistical_features)
        
        return result
    
    def get_feature_names(self, input_features: List[str]) -> List[str]:
        """Get names of generated features."""
        return self.generated_feature_names


class CategoricalFeatureGenerator(FeatureGenerator):
    """
    Generator for categorical features including frequency encoding, target encoding, and combinations.
    """
    
    def __init__(self, include_frequency: bool = True, include_combinations: bool = True,
                 max_combinations: int = 5):
        """
        Initialize categorical feature generator.
        
        Args:
            include_frequency: Whether to include frequency encoding
            include_combinations: Whether to include categorical combinations
            max_combinations: Maximum number of categorical combinations
        """
        super().__init__('categorical_feature_generator')
        self.include_frequency = include_frequency
        self.include_combinations = include_combinations
        self.max_combinations = max_combinations
        self.generated_feature_names = []
        self.frequency_maps = {}
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate categorical features."""
        result = data.copy()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_columns:
            return result
        
        self.generated_feature_names = []
        
        # Frequency encoding
        if self.include_frequency:
            result = self._add_frequency_features(result, categorical_columns)
        
        # Categorical combinations
        if self.include_combinations and len(categorical_columns) > 1:
            result = self._add_combination_features(result, categorical_columns)
        
        return result
    
    def _add_frequency_features(self, data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Add frequency encoding features."""
        result = data.copy()
        
        for column in categorical_columns:
            # Calculate frequency of each category
            frequency_map = data[column].value_counts().to_dict()
            self.frequency_maps[column] = frequency_map
            
            # Create frequency feature
            feature_name = f"{column}_frequency"
            result[feature_name] = data[column].map(frequency_map).fillna(0)
            self.generated_feature_names.append(feature_name)
        
        return result
    
    def _add_combination_features(self, data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Add categorical combination features."""
        result = data.copy()
        combinations_added = 0
        
        # Generate pairwise combinations
        for col1, col2 in itertools.combinations(categorical_columns, 2):
            if combinations_added >= self.max_combinations:
                break
            
            # Create combined categorical feature and encode it numerically
            feature_name = f"{col1}_{col2}_combo"
            combined_values = data[col1].astype(str) + "_" + data[col2].astype(str)
            
            # Encode the combined values as integers
            unique_combinations = combined_values.unique()
            combo_map = {combo: i for i, combo in enumerate(unique_combinations)}
            result[feature_name] = combined_values.map(combo_map)
            
            self.generated_feature_names.append(feature_name)
            combinations_added += 1
        
        return result
    
    def get_feature_names(self, input_features: List[str]) -> List[str]:
        """Get names of generated features."""
        return self.generated_feature_names


class TextFeatureGenerator(FeatureGenerator):
    """
    Generator for text features including length, word count, and character statistics.
    """
    
    def __init__(self, include_length: bool = True, include_word_stats: bool = True,
                 include_char_stats: bool = True):
        """
        Initialize text feature generator.
        
        Args:
            include_length: Whether to include text length features
            include_word_stats: Whether to include word statistics
            include_char_stats: Whether to include character statistics
        """
        super().__init__('text_feature_generator')
        self.include_length = include_length
        self.include_word_stats = include_word_stats
        self.include_char_stats = include_char_stats
        self.generated_feature_names = []
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate text features."""
        result = data.copy()
        text_columns = []
        
        # Identify text columns (object columns with long average length)
        for column in data.select_dtypes(include=['object']).columns:
            if data[column].astype(str).str.len().mean() > 20:  # Threshold for text
                text_columns.append(column)
        
        if not text_columns:
            return result
        
        self.generated_feature_names = []
        
        for column in text_columns:
            if self.include_length:
                result = self._add_length_features(result, column)
            
            if self.include_word_stats:
                result = self._add_word_features(result, column)
            
            if self.include_char_stats:
                result = self._add_character_features(result, column)
        
        return result
    
    def _add_length_features(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Add text length features."""
        result = data.copy()
        text_series = data[column].astype(str)
        
        # Character length
        feature_name = f"{column}_char_length"
        result[feature_name] = text_series.str.len()
        self.generated_feature_names.append(feature_name)
        
        # Word count
        feature_name = f"{column}_word_count"
        result[feature_name] = text_series.str.split().str.len()
        self.generated_feature_names.append(feature_name)
        
        # Average word length
        feature_name = f"{column}_avg_word_length"
        result[feature_name] = result[f"{column}_char_length"] / (result[f"{column}_word_count"] + 1e-8)
        self.generated_feature_names.append(feature_name)
        
        return result
    
    def _add_word_features(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Add word-based features."""
        result = data.copy()
        text_series = data[column].astype(str)
        
        # Sentence count (approximate)
        feature_name = f"{column}_sentence_count"
        result[feature_name] = text_series.str.count(r'[.!?]+')
        self.generated_feature_names.append(feature_name)
        
        # Unique word ratio
        feature_name = f"{column}_unique_word_ratio"
        word_counts = text_series.str.split().str.len()
        unique_word_counts = text_series.str.split().apply(lambda x: len(set(x)) if x else 0)
        result[feature_name] = unique_word_counts / (word_counts + 1e-8)
        self.generated_feature_names.append(feature_name)
        
        return result
    
    def _add_character_features(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Add character-based features."""
        result = data.copy()
        text_series = data[column].astype(str)
        
        # Uppercase ratio
        feature_name = f"{column}_uppercase_ratio"
        result[feature_name] = text_series.str.count(r'[A-Z]') / (text_series.str.len() + 1e-8)
        self.generated_feature_names.append(feature_name)
        
        # Digit ratio
        feature_name = f"{column}_digit_ratio"
        result[feature_name] = text_series.str.count(r'\d') / (text_series.str.len() + 1e-8)
        self.generated_feature_names.append(feature_name)
        
        # Punctuation ratio
        feature_name = f"{column}_punctuation_ratio"
        result[feature_name] = text_series.str.count(r'[^\w\s]') / (text_series.str.len() + 1e-8)
        self.generated_feature_names.append(feature_name)
        
        return result
    
    def get_feature_names(self, input_features: List[str]) -> List[str]:
        """Get names of generated features."""
        return self.generated_feature_names


class FeatureSelector:
    """
    Feature selection component that selects the most important features.
    """
    
    def __init__(self, method: str = 'auto', k: int = 10, task_type: str = 'classification'):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('univariate', 'rfe', 'model_based', 'auto')
            k: Number of features to select
            task_type: Type of ML task ('classification' or 'regression')
        """
        self.method = method
        self.k = k
        self.task_type = task_type
        self.selector = None
        self.selected_features = []
        self.feature_scores = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """Fit the feature selector."""
        if self.method == 'auto':
            # Choose method based on data characteristics
            if X.shape[1] > 100:
                method = 'univariate'  # Fast for high-dimensional data
            elif X.shape[0] < 1000:
                method = 'model_based'  # Good for small datasets
            else:
                method = 'rfe'  # Balanced approach
        else:
            method = self.method
        
        if method == 'univariate':
            self._fit_univariate(X, y)
        elif method == 'rfe':
            self._fit_rfe(X, y)
        elif method == 'model_based':
            self._fit_model_based(X, y)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features."""
        if not self.selected_features:
            return X
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(X, y).transform(X)
    
    def _fit_univariate(self, X: pd.DataFrame, y: pd.Series):
        """Fit univariate feature selection."""
        if self.task_type == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression
        
        self.selector = SelectKBest(score_func=score_func, k=min(self.k, X.shape[1]))
        self.selector.fit(X, y)
        
        # Get selected features
        selected_mask = self.selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        # Store feature scores
        scores = self.selector.scores_
        for i, feature in enumerate(X.columns):
            self.feature_scores[feature] = scores[i] if not np.isnan(scores[i]) else 0
    
    def _fit_rfe(self, X: pd.DataFrame, y: pd.Series):
        """Fit recursive feature elimination."""
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        self.selector = RFE(estimator=estimator, n_features_to_select=min(self.k, X.shape[1]))
        self.selector.fit(X, y)
        
        # Get selected features
        selected_mask = self.selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        # Store feature rankings as scores (lower rank = higher score)
        rankings = self.selector.ranking_
        for i, feature in enumerate(X.columns):
            self.feature_scores[feature] = 1.0 / rankings[i]  # Convert rank to score
    
    def _fit_model_based(self, X: pd.DataFrame, y: pd.Series):
        """Fit model-based feature selection."""
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        estimator.fit(X, y)
        
        # Use feature importances for selection
        importances = estimator.feature_importances_
        
        # Select top k features
        feature_importance_pairs = list(zip(X.columns, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        self.selected_features = [pair[0] for pair in feature_importance_pairs[:self.k]]
        
        # Store feature scores
        for feature, importance in feature_importance_pairs:
            self.feature_scores[feature] = importance
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_scores


class FeatureEngineer:
    """
    Main feature engineering component that orchestrates feature generation and selection.
    """
    
    def __init__(self, enable_generation: bool = True, enable_selection: bool = True,
                 max_features: int = 50, task_type: str = 'classification'):
        """
        Initialize feature engineer.
        
        Args:
            enable_generation: Whether to enable feature generation
            enable_selection: Whether to enable feature selection
            max_features: Maximum number of features to keep after selection
            task_type: Type of ML task ('classification' or 'regression')
        """
        self.enable_generation = enable_generation
        self.enable_selection = enable_selection
        self.max_features = max_features
        self.task_type = task_type
        
        # Feature generators
        self.numerical_generator = NumericalFeatureGenerator()
        self.categorical_generator = CategoricalFeatureGenerator()
        self.text_generator = TextFeatureGenerator()
        
        # Feature selector
        self.feature_selector = FeatureSelector(k=max_features, task_type=task_type)
        
        # State
        self.fitted = False
        self.original_features = []
        self.generated_features = []
        self.selected_features = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """Fit the feature engineer."""
        self.original_features = X.columns.tolist()
        result = X.copy()
        
        # Feature generation
        if self.enable_generation:
            # Generate numerical features
            result = self.numerical_generator.generate_features(result)
            
            # Generate categorical features
            result = self.categorical_generator.generate_features(result)
            
            # Generate text features
            result = self.text_generator.generate_features(result)
            
            # Track generated features
            self.generated_features = [col for col in result.columns if col not in self.original_features]
        
        # Feature selection
        if self.enable_selection and y is not None:
            # Only select from numeric columns for feature selection
            numeric_result = result.select_dtypes(include=[np.number])
            if len(numeric_result.columns) > 0:
                self.feature_selector.fit(numeric_result, y)
                self.selected_features = self.feature_selector.selected_features
            else:
                # If no numeric columns, keep all columns
                self.selected_features = result.columns.tolist()
        else:
            self.selected_features = result.columns.tolist()
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by applying feature engineering."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        result = X.copy()
        
        # Apply feature generation
        if self.enable_generation:
            result = self.numerical_generator.generate_features(result)
            result = self.categorical_generator.generate_features(result)
            result = self.text_generator.generate_features(result)
        
        # Apply feature selection
        if self.enable_selection and self.selected_features:
            # Only select features that exist in the current data
            available_features = [f for f in self.selected_features if f in result.columns]
            result = result[available_features]
        
        return result
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        return self.selected_features if self.selected_features else []
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.enable_selection:
            return self.feature_selector.get_feature_scores()
        return {}
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of feature generation."""
        return {
            'original_features': len(self.original_features),
            'generated_features': len(self.generated_features),
            'total_features_before_selection': len(self.original_features) + len(self.generated_features),
            'selected_features': len(self.selected_features),
            'feature_reduction_ratio': len(self.selected_features) / (len(self.original_features) + len(self.generated_features)) if self.generated_features else 1.0
        }


# Update the PreprocessingPipeline to include feature engineering
class FeatureEngineeringTransformer(PreprocessingTransformer):
    """
    Transformer wrapper for FeatureEngineer to integrate with preprocessing pipeline.
    """
    
    def __init__(self, enable_generation: bool = True, enable_selection: bool = True,
                 max_features: int = 50, task_type: str = 'classification'):
        """Initialize feature engineering transformer."""
        super().__init__('feature_engineering')
        self.feature_engineer = FeatureEngineer(
            enable_generation=enable_generation,
            enable_selection=enable_selection,
            max_features=max_features,
            task_type=task_type
        )
        self.target_column = None
    
    def fit(self, data: pd.DataFrame, target_column: Optional[str] = None) -> 'FeatureEngineeringTransformer':
        """Fit the feature engineering transformer."""
        self.target_column = target_column
        
        if target_column and target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            self.feature_engineer.fit(X, y)
        else:
            self.feature_engineer.fit(data)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using feature engineering."""
        if not self.fitted:
            raise ValueError("FeatureEngineeringTransformer must be fitted before transform")
        
        if self.target_column and self.target_column in data.columns:
            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]
            
            X_transformed = self.feature_engineer.transform(X)
            
            # Add target column back
            result = X_transformed.copy()
            result[self.target_column] = y.reset_index(drop=True)
            return result
        else:
            return self.feature_engineer.transform(data)
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {
            'name': self.name,
            'enable_generation': self.feature_engineer.enable_generation,
            'enable_selection': self.feature_engineer.enable_selection,
            'max_features': self.feature_engineer.max_features,
            'task_type': self.feature_engineer.task_type
        }