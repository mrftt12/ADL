#!/usr/bin/env python3
"""
Demonstration of the AutoML Framework Evaluation Service

This script shows how to use the comprehensive evaluation and model comparison
functionality provided by the evaluation service.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from automl_framework.services.evaluation_service import (
    ComprehensiveEvaluator, ModelComparator
)
from automl_framework.models.data_models import TaskType, PerformanceMetrics


def demo_classification_evaluation():
    """Demonstrate classification model evaluation."""
    print("=" * 60)
    print("CLASSIFICATION EVALUATION DEMO")
    print("=" * 60)
    
    # Create sample classification data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=3, 
        n_informative=15, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Initialize evaluator and comparator
    evaluator = ComprehensiveEvaluator(
        TaskType.CLASSIFICATION, 
        class_labels=['Class_0', 'Class_1', 'Class_2']
    )
    comparator = ModelComparator(TaskType.CLASSIFICATION)
    
    # Train models and collect results
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Evaluate model
        metrics = evaluator.evaluate_classification(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)
        model_results[name] = metrics
        
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"  F1 Score: {metrics.f1_score:.4f}")
        print(f"  AUC-ROC: {metrics.additional_metrics.get('auc_roc', 'N/A'):.4f}")
    
    # Compare models
    print("\n" + "=" * 40)
    print("MODEL COMPARISON")
    print("=" * 40)
    
    comparison_result = comparator.compare_models(model_results)
    
    print(f"\nBest Model: {comparison_result.best_model}")
    print("\nModel Rankings:")
    for ranking in comparison_result.model_rankings:
        print(f"  {ranking['rank']}. {ranking['model_name']} (Score: {ranking['composite_score']:.4f})")
    
    # Generate confusion matrix for best model
    best_model_name = comparison_result.best_model
    best_model = trained_models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    
    cm_result = evaluator.generate_confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion Matrix for {best_model_name}:")
    print(cm_result.matrix)
    
    # Cross-validation comparison
    print("\n" + "=" * 40)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 40)
    
    cv_results = comparator.compare_models_with_cv(
        trained_models, X_train, y_train, cv_folds=5, scoring='accuracy'
    )
    
    for model_name, cv_result in cv_results.items():
        print(f"\n{model_name}:")
        print(f"  Mean CV Score: {cv_result.mean_score:.4f} ± {cv_result.std_score:.4f}")
        print(f"  95% CI: [{cv_result.confidence_interval[0]:.4f}, {cv_result.confidence_interval[1]:.4f}]")
    
    # Statistical significance tests
    statistical_tests = comparator.perform_model_significance_tests(cv_results)
    print("\n" + "=" * 40)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 40)
    
    for test_name, test_result in statistical_tests.items():
        print(f"\n{test_name}:")
        print(f"  {test_result.interpretation}")
    
    # Generate performance report
    print("\n" + "=" * 60)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 60)
    
    report = comparator.generate_performance_report(comparison_result, cv_results)
    print(report)


def demo_regression_evaluation():
    """Demonstrate regression model evaluation."""
    print("\n\n" + "=" * 60)
    print("REGRESSION EVALUATION DEMO")
    print("=" * 60)
    
    # Create sample regression data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train multiple models
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    # Initialize evaluator and comparator
    evaluator = ComprehensiveEvaluator(TaskType.REGRESSION)
    comparator = ModelComparator(TaskType.REGRESSION)
    
    # Train models and collect results
    model_results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        metrics = evaluator.evaluate_regression(y_true=y_test, y_pred=y_pred)
        model_results[name] = metrics
        
        print(f"  MSE: {metrics.loss:.4f}")
        print(f"  R² Score: {metrics.additional_metrics.get('r2_score', 'N/A'):.4f}")
        print(f"  MAE: {metrics.additional_metrics.get('mae', 'N/A'):.4f}")
    
    # Compare models
    print("\n" + "=" * 40)
    print("MODEL COMPARISON")
    print("=" * 40)
    
    comparison_result = comparator.compare_models(model_results)
    
    print(f"\nBest Model: {comparison_result.best_model}")
    print("\nModel Rankings:")
    for ranking in comparison_result.model_rankings:
        print(f"  {ranking['rank']}. {ranking['model_name']} (Score: {ranking['composite_score']:.4f})")


def demo_bootstrap_evaluation():
    """Demonstrate bootstrap model evaluation."""
    print("\n\n" + "=" * 60)
    print("BOOTSTRAP EVALUATION DEMO")
    print("=" * 60)
    
    # Create sample data
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    
    # Initialize comparator
    comparator = ModelComparator(TaskType.CLASSIFICATION)
    
    # Create and evaluate model with bootstrap
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    print("Performing bootstrap evaluation (this may take a moment)...")
    bootstrap_result = comparator.bootstrap_model_evaluation(
        model, X, y, n_bootstrap=100, random_state=42
    )
    
    print(f"\nBootstrap Results:")
    print(f"  Mean Score: {bootstrap_result['mean_score']:.4f}")
    print(f"  Std Score: {bootstrap_result['std_score']:.4f}")
    print(f"  Confidence Intervals:")
    for level, (lower, upper) in bootstrap_result['confidence_intervals'].items():
        print(f"    {level}: [{lower:.4f}, {upper:.4f}]")


if __name__ == "__main__":
    print("AutoML Framework - Evaluation Service Demo")
    print("This demo showcases the comprehensive evaluation capabilities")
    print("including model comparison, statistical testing, and bootstrap evaluation.\n")
    
    try:
        # Run classification demo
        demo_classification_evaluation()
        
        # Run regression demo
        demo_regression_evaluation()
        
        # Run bootstrap demo
        demo_bootstrap_evaluation()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe evaluation service provides:")
        print("• Comprehensive metrics for classification and regression")
        print("• Confusion matrix generation and visualization")
        print("• Statistical significance testing between models")
        print("• Cross-validation evaluation")
        print("• Bootstrap evaluation for confidence intervals")
        print("• Model comparison and ranking")
        print("• Performance reporting and visualization")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()