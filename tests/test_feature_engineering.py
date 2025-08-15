"""
Unit tests for feature engineering components.

Tests feature generation, selection, and the main FeatureEngineer class.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from automl_framework.services.data_processing import (
    NumericalFeatureGenerator, CategoricalFeatureGenerator, TextFeatureGenerator,
    FeatureSelector, FeatureEngineer, FeatureEngineeringTransformer
)


class TestNumericalFeatureGenerator:
    """Test suite for NumericalFeatureGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = NumericalFeatureGenerator()
        assert generator.name == 'numerical_feature_generator'
        assert generator.include_polynomial == True
        assert generator.include_interactions == True
        assert generator.include_statistical == True
    
    def test_polynomial_features(self):
        """Test polynomial feature generation."""
        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [2, 4, 6, 8, 10],
            'cat': ['A', 'B', 'C', 'D', 'E']
        })
        
        generator = NumericalFeatureGenerator(
            include_polynomial=True,
            include_interactions=False,
            include_statistical=False,
            polynomial_degree=2
        )
        
        result = generator.generate_features(df)
        
        # Should have original features plus polynomial features
        assert 'num1_poly_2' in result.columns
        assert 'num2_poly_2' in result.columns
        assert result['num1_poly_2'].iloc[0] == 1  # 1^2 = 1
        assert result['num1_poly_2'].iloc[1] == 4  # 2^2 = 4
    
    def test_interaction_features(self):
        """Test interaction feature generation."""
        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [2, 4, 6, 8, 10]
        })
        
        generator = NumericalFeatureGenerator(
            include_polynomial=False,
            include_interactions=True,
            include_statistical=False
        )
        
        result = generator.generate_features(df)
        
        # Should have interaction features
        assert 'num1_x_num2' in result.columns
        assert 'num1_div_num2' in result.columns
        assert result['num1_x_num2'].iloc[0] == 2  # 1 * 2 = 2
        assert result['num1_div_num2'].iloc[0] == pytest.approx(0.5, abs=1e-6)  # 1 / 2 = 0.5
    
    def test_statistical_features(self):
        """Test statistical feature generation."""
        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [2, 4, 6, 8, 10],
            'num3': [1, 1, 1, 1, 1]
        })
        
        generator = NumericalFeatureGenerator(
            include_polynomial=False,
            include_interactions=False,
            include_statistical=True
        )
        
        result = generator.generate_features(df)
        
        # Should have statistical features
        assert 'row_mean' in result.columns
        assert 'row_std' in result.columns
        assert 'row_min' in result.columns
        assert 'row_max' in result.columns
        assert 'row_range' in result.columns
        
        # Check first row: [1, 2, 1]
        assert result['row_mean'].iloc[0] == pytest.approx(4/3, abs=1e-6)
        assert result['row_min'].iloc[0] == 1
        assert result['row_max'].iloc[0] == 2
        assert result['row_range'].iloc[0] == 1
    
    def test_no_numeric_columns(self):
        """Test behavior with no numeric columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        
        generator = NumericalFeatureGenerator()
        result = generator.generate_features(df)
        
        # Should return original dataframe unchanged
        pd.testing.assert_frame_equal(result, df)


class TestCategoricalFeatureGenerator:
    """Test suite for CategoricalFeatureGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = CategoricalFeatureGenerator()
        assert generator.name == 'categorical_feature_generator'
        assert generator.include_frequency == True
        assert generator.include_combinations == True
    
    def test_frequency_features(self):
        """Test frequency encoding features."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'C', 'A'],
            'cat2': ['X', 'Y', 'X', 'X', 'Y'],
            'num': [1, 2, 3, 4, 5]
        })
        
        generator = CategoricalFeatureGenerator(
            include_frequency=True,
            include_combinations=False
        )
        
        result = generator.generate_features(df)
        
        # Should have frequency features
        assert 'cat1_frequency' in result.columns
        assert 'cat2_frequency' in result.columns
        
        # Check frequencies: A appears 3 times, B appears 1 time, C appears 1 time
        assert result['cat1_frequency'].iloc[0] == 3  # A
        assert result['cat1_frequency'].iloc[1] == 1  # B
        assert result['cat1_frequency'].iloc[3] == 1  # C
        
        # Check frequencies: X appears 3 times, Y appears 2 times
        assert result['cat2_frequency'].iloc[0] == 3  # X
        assert result['cat2_frequency'].iloc[1] == 2  # Y
    
    def test_combination_features(self):
        """Test categorical combination features."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'A'],
            'cat2': ['X', 'Y', 'X']
        })
        
        generator = CategoricalFeatureGenerator(
            include_frequency=False,
            include_combinations=True
        )
        
        result = generator.generate_features(df)
        
        # Should have combination feature (now numeric)
        assert 'cat1_cat2_combo' in result.columns
        # Values should be numeric encodings of combinations
        assert pd.api.types.is_numeric_dtype(result['cat1_cat2_combo'])
        # First and third rows should have same value (both A_X)
        assert result['cat1_cat2_combo'].iloc[0] == result['cat1_cat2_combo'].iloc[2]
        # Second row should have different value (B_Y)
        assert result['cat1_cat2_combo'].iloc[1] != result['cat1_cat2_combo'].iloc[0]
    
    def test_no_categorical_columns(self):
        """Test behavior with no categorical columns."""
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4, 5, 6]
        })
        
        generator = CategoricalFeatureGenerator()
        result = generator.generate_features(df)
        
        # Should return original dataframe unchanged
        pd.testing.assert_frame_equal(result, df)


class TestTextFeatureGenerator:
    """Test suite for TextFeatureGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = TextFeatureGenerator()
        assert generator.name == 'text_feature_generator'
        assert generator.include_length == True
        assert generator.include_word_stats == True
        assert generator.include_char_stats == True
    
    def test_length_features(self):
        """Test text length features."""
        df = pd.DataFrame({
            'text': ['This is a sample text for testing purposes',
                    'Short text',
                    'Another longer text with more words and characters'],
            'num': [1, 2, 3]
        })
        
        generator = TextFeatureGenerator(
            include_length=True,
            include_word_stats=False,
            include_char_stats=False
        )
        
        result = generator.generate_features(df)
        
        # Should have length features
        assert 'text_char_length' in result.columns
        assert 'text_word_count' in result.columns
        assert 'text_avg_word_length' in result.columns
        
        # Check first text: 'This is a sample text for testing purposes'
        assert result['text_char_length'].iloc[0] == 42
        assert result['text_word_count'].iloc[0] == 8
        assert result['text_avg_word_length'].iloc[0] == pytest.approx(42/8, abs=1e-6)
    
    def test_word_features(self):
        """Test word-based features."""
        df = pd.DataFrame({
            'text': ['This is a test. Another sentence!',
                    'Single sentence.',
                    'Word word word unique different']
        })
        
        generator = TextFeatureGenerator(
            include_length=False,
            include_word_stats=True,
            include_char_stats=False
        )
        
        result = generator.generate_features(df)
        
        # Should have word features
        assert 'text_sentence_count' in result.columns
        assert 'text_unique_word_ratio' in result.columns
        
        # Check sentence count (periods, exclamations)
        assert result['text_sentence_count'].iloc[0] == 2  # Two sentences
        assert result['text_sentence_count'].iloc[1] == 1  # One sentence
    
    def test_character_features(self):
        """Test character-based features."""
        df = pd.DataFrame({
            'text': ['This Has UPPERCASE and 123 numbers!',
                    'lowercase only text',
                    'Mixed Case With Punctuation!!!']
        })
        
        generator = TextFeatureGenerator(
            include_length=False,
            include_word_stats=False,
            include_char_stats=True
        )
        
        result = generator.generate_features(df)
        
        # Should have character features
        assert 'text_uppercase_ratio' in result.columns
        assert 'text_digit_ratio' in result.columns
        assert 'text_punctuation_ratio' in result.columns
        
        # Check first text has uppercase letters
        assert result['text_uppercase_ratio'].iloc[0] > 0
        # Check first text has digits
        assert result['text_digit_ratio'].iloc[0] > 0
        # Check first text has punctuation
        assert result['text_punctuation_ratio'].iloc[0] > 0
    
    def test_no_text_columns(self):
        """Test behavior with no text columns."""
        df = pd.DataFrame({
            'short': ['A', 'B', 'C'],  # Too short to be considered text
            'num': [1, 2, 3]
        })
        
        generator = TextFeatureGenerator()
        result = generator.generate_features(df)
        
        # Should return original dataframe unchanged
        pd.testing.assert_frame_equal(result, df)


class TestFeatureSelector:
    """Test suite for FeatureSelector."""
    
    def setup_method(self):
        """Set up test data."""
        # Create synthetic classification data
        X_class, y_class = make_classification(
            n_samples=100, n_features=20, n_informative=10, n_redundant=5, random_state=42
        )
        self.X_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(20)])
        self.y_class = pd.Series(y_class)
        
        # Create synthetic regression data
        X_reg, y_reg = make_regression(
            n_samples=100, n_features=20, n_informative=10, noise=0.1, random_state=42
        )
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(20)])
        self.y_reg = pd.Series(y_reg)
    
    def test_init(self):
        """Test selector initialization."""
        selector = FeatureSelector()
        assert selector.method == 'auto'
        assert selector.k == 10
        assert selector.task_type == 'classification'
    
    def test_univariate_selection_classification(self):
        """Test univariate feature selection for classification."""
        selector = FeatureSelector(method='univariate', k=5, task_type='classification')
        selector.fit(self.X_class, self.y_class)
        
        assert len(selector.selected_features) == 5
        assert len(selector.feature_scores) == 20
        
        # Transform data
        X_selected = selector.transform(self.X_class)
        assert X_selected.shape[1] == 5
    
    def test_univariate_selection_regression(self):
        """Test univariate feature selection for regression."""
        selector = FeatureSelector(method='univariate', k=5, task_type='regression')
        selector.fit(self.X_reg, self.y_reg)
        
        assert len(selector.selected_features) == 5
        assert len(selector.feature_scores) == 20
        
        # Transform data
        X_selected = selector.transform(self.X_reg)
        assert X_selected.shape[1] == 5
    
    def test_rfe_selection(self):
        """Test RFE feature selection."""
        selector = FeatureSelector(method='rfe', k=5, task_type='classification')
        selector.fit(self.X_class, self.y_class)
        
        assert len(selector.selected_features) == 5
        assert len(selector.feature_scores) == 20
        
        # Transform data
        X_selected = selector.transform(self.X_class)
        assert X_selected.shape[1] == 5
    
    def test_model_based_selection(self):
        """Test model-based feature selection."""
        selector = FeatureSelector(method='model_based', k=5, task_type='classification')
        selector.fit(self.X_class, self.y_class)
        
        assert len(selector.selected_features) == 5
        assert len(selector.feature_scores) == 20
        
        # Transform data
        X_selected = selector.transform(self.X_class)
        assert X_selected.shape[1] == 5
    
    def test_auto_method_selection(self):
        """Test automatic method selection."""
        # Small dataset should use model_based
        small_X = self.X_class.iloc[:50]
        small_y = self.y_class.iloc[:50]
        
        selector = FeatureSelector(method='auto', k=5)
        selector.fit(small_X, small_y)
        
        assert len(selector.selected_features) == 5
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        selector = FeatureSelector(method='univariate', k=5)
        X_selected = selector.fit_transform(self.X_class, self.y_class)
        
        assert X_selected.shape[1] == 5
        assert len(selector.selected_features) == 5


class TestFeatureEngineer:
    """Test suite for FeatureEngineer."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [2, 4, 6, 8, 10],
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'Z'],
            'text': ['This is sample text for testing',
                    'Another text sample here',
                    'Short text',
                    'Longer text with more words and content',
                    'Final text sample'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_init(self):
        """Test engineer initialization."""
        engineer = FeatureEngineer()
        assert engineer.enable_generation == True
        assert engineer.enable_selection == True
        assert engineer.max_features == 50
        assert engineer.task_type == 'classification'
    
    def test_feature_generation_only(self):
        """Test feature generation without selection."""
        X = self.df.drop(columns=['target'])
        
        engineer = FeatureEngineer(enable_generation=True, enable_selection=False)
        result = engineer.fit_transform(X)
        
        # Should have more features than original
        assert result.shape[1] > X.shape[1]
        assert len(engineer.generated_features) > 0
        assert engineer.selected_features == result.columns.tolist()
    
    def test_feature_selection_only(self):
        """Test feature selection without generation."""
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        
        engineer = FeatureEngineer(enable_generation=False, enable_selection=True, max_features=3)
        result = engineer.fit_transform(X, y)
        
        # Should have fewer or equal features than original
        assert result.shape[1] <= min(3, X.shape[1])
        assert len(engineer.generated_features) == 0
        assert len(engineer.selected_features) <= 3
    
    def test_full_feature_engineering(self):
        """Test complete feature engineering pipeline."""
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        
        engineer = FeatureEngineer(enable_generation=True, enable_selection=True, max_features=10)
        result = engineer.fit_transform(X, y)
        
        # Should have generated and then selected features
        assert len(engineer.generated_features) > 0
        assert len(engineer.selected_features) <= 10
        assert result.shape[1] == len(engineer.selected_features)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        
        engineer = FeatureEngineer(enable_selection=True)
        engineer.fit(X, y)
        
        importance = engineer.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
    
    def test_generation_summary(self):
        """Test generation summary."""
        X = self.df.drop(columns=['target'])
        y = self.df['target']
        
        engineer = FeatureEngineer()
        engineer.fit(X, y)
        
        summary = engineer.get_generation_summary()
        assert 'original_features' in summary
        assert 'generated_features' in summary
        assert 'selected_features' in summary
        assert 'feature_reduction_ratio' in summary
        
        assert summary['original_features'] == X.shape[1]
        assert summary['generated_features'] > 0
    
    def test_transform_without_fit(self):
        """Test error when transforming without fitting."""
        X = self.df.drop(columns=['target'])
        
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="must be fitted before transform"):
            engineer.transform(X)
    
    def test_no_target_provided(self):
        """Test behavior when no target is provided."""
        X = self.df.drop(columns=['target'])
        
        engineer = FeatureEngineer(enable_selection=True)
        result = engineer.fit_transform(X)  # No y provided
        
        # Should still generate features but not select
        assert result.shape[1] > X.shape[1]  # Generated features
        assert len(engineer.generated_features) > 0


class TestFeatureEngineeringTransformer:
    """Test suite for FeatureEngineeringTransformer."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [2, 4, 6, 8, 10],
            'cat': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_init(self):
        """Test transformer initialization."""
        transformer = FeatureEngineeringTransformer()
        assert transformer.name == 'feature_engineering'
        assert transformer.feature_engineer is not None
    
    def test_fit_transform_with_target(self):
        """Test fit and transform with target column."""
        transformer = FeatureEngineeringTransformer(max_features=5)
        result = transformer.fit(self.df, target_column='target').transform(self.df)
        
        # Should have target column preserved
        assert 'target' in result.columns
        # Should have applied feature engineering to other columns
        assert result.shape[1] <= 6  # max_features + target
    
    def test_fit_transform_without_target(self):
        """Test fit and transform without target column."""
        df_no_target = self.df.drop(columns=['target'])
        
        transformer = FeatureEngineeringTransformer()
        result = transformer.fit_transform(df_no_target)
        
        # Should have more features due to generation
        assert result.shape[1] > df_no_target.shape[1]
    
    def test_get_params(self):
        """Test parameter extraction."""
        transformer = FeatureEngineeringTransformer(
            enable_generation=False,
            max_features=20,
            task_type='regression'
        )
        
        params = transformer.get_params()
        assert params['name'] == 'feature_engineering'
        assert params['enable_generation'] == False
        assert params['max_features'] == 20
        assert params['task_type'] == 'regression'
    
    def test_transform_before_fit(self):
        """Test error when transforming before fitting."""
        transformer = FeatureEngineeringTransformer()
        
        with pytest.raises(ValueError, match="must be fitted before transform"):
            transformer.transform(self.df)


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering components."""
    
    def test_end_to_end_pipeline(self):
        """Test complete feature engineering pipeline."""
        # Create more complex dataset
        np.random.seed(42)
        df = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'num3': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
            'cat2': np.random.choice(['X', 'Y'], 100),
            'text': ['Sample text ' + str(i) for i in range(100)],
            'target': np.random.choice([0, 1], 100)
        })
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Apply feature engineering
        engineer = FeatureEngineer(
            enable_generation=True,
            enable_selection=True,
            max_features=15,
            task_type='classification'
        )
        
        X_engineered = engineer.fit_transform(X, y)
        
        # Verify results
        assert X_engineered.shape[0] == 100  # Same number of samples
        assert X_engineered.shape[1] <= 15  # Respects max_features
        assert len(engineer.generated_features) > 0  # Features were generated
        assert len(engineer.selected_features) > 0  # Features were selected
        
        # Check that we can transform new data
        X_new = X.iloc[:10]  # First 10 samples
        X_new_engineered = engineer.transform(X_new)
        assert X_new_engineered.shape[1] == X_engineered.shape[1]  # Same number of features
    
    def test_feature_engineering_with_missing_values(self):
        """Test feature engineering with missing values."""
        df = pd.DataFrame({
            'num1': [1, 2, np.nan, 4, 5],
            'num2': [2, np.nan, 6, 8, 10],
            'cat': ['A', 'B', None, 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        engineer = FeatureEngineer(max_features=10)
        result = engineer.fit_transform(X, y)
        
        # Should handle missing values gracefully
        assert result.shape[0] == 5
        assert result.shape[1] <= 10
    
    def test_feature_engineering_performance(self):
        """Test feature engineering on larger dataset."""
        # Create larger dataset
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'num1': np.random.randn(n_samples),
            'num2': np.random.randn(n_samples),
            'cat1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'cat2': np.random.choice(['X', 'Y', 'Z'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        engineer = FeatureEngineer(max_features=20)
        
        # Should complete without errors
        result = engineer.fit_transform(X, y)
        
        assert result.shape[0] == n_samples
        assert result.shape[1] <= 20