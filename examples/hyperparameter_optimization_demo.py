#!/usr/bin/env python3
"""
Demonstration of the hyperparameter optimization service
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from automl_framework.services.hyperparameter_optimization import (
    HyperparameterOptimizationService,
    create_default_hyperparameter_space
)
from automl_framework.core.interfaces import Architecture


def mock_training_function(params):
    """
    Mock training function that simulates model training
    Returns higher scores for better hyperparameter combinations
    """
    lr = params.get("learning_rate", 0.001)
    batch_size = params.get("batch_size", 32)
    optimizer = params.get("optimizer", "adam")
    weight_decay = params.get("weight_decay", 1e-4)
    dropout_rate = params.get("dropout_rate", 0.1)
    
    # Simulate optimal values
    optimal_lr = 0.01
    optimal_batch_size = 64
    optimal_optimizer = "adam"
    optimal_weight_decay = 1e-4
    optimal_dropout = 0.2
    
    # Calculate penalties for deviation from optimal values
    lr_penalty = abs(np.log10(lr) - np.log10(optimal_lr)) / 2.0
    batch_penalty = abs(batch_size - optimal_batch_size) / optimal_batch_size
    optimizer_penalty = 0.0 if optimizer == optimal_optimizer else 0.1
    wd_penalty = abs(np.log10(weight_decay) - np.log10(optimal_weight_decay)) / 2.0
    dropout_penalty = abs(dropout_rate - optimal_dropout) / optimal_dropout
    
    # Add some noise to simulate real training variability
    noise = np.random.normal(0, 0.02)
    
    # Calculate final score (higher is better)
    score = 1.0 - (lr_penalty + batch_penalty + optimizer_penalty + wd_penalty + dropout_penalty) / 5.0 + noise
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def demonstrate_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization with both algorithms"""
    
    print("=== Hyperparameter Optimization Demo ===\n")
    
    # Create a mock architecture
    architecture = Architecture(
        id="demo_arch",
        layers=[],
        connections=[],
        input_shape=(224, 224, 3),
        output_shape=(10,),
        parameter_count=1000000,
        flops=2000000
    )
    
    print("Architecture:", architecture.id)
    print("Input shape:", architecture.input_shape)
    print("Output shape:", architecture.output_shape)
    print("Parameters:", architecture.parameter_count)
    print()
    
    # Test both algorithms
    algorithms = ["bayesian", "tpe"]
    results = {}
    
    for algorithm in algorithms:
        print(f"--- Testing {algorithm.upper()} Optimization ---")
        
        # Initialize service
        service = HyperparameterOptimizationService(algorithm=algorithm)
        
        # Define search space
        search_space = service.define_search_space(architecture)
        print(f"Search space parameters: {list(search_space.parameters.keys())}")
        
        # Run optimization
        print(f"Running {algorithm} optimization...")
        best_config = service.optimize(
            objective_function=mock_training_function,
            search_space=search_space,
            max_trials=20
        )
        
        # Get optimization history
        history = service.get_optimization_history()
        
        # Store results
        results[algorithm] = {
            'config': best_config,
            'history': history,
            'best_score': max(t.metrics.accuracy for t in history if t.metrics)
        }
        
        print(f"Best configuration found:")
        print(f"  Learning rate: {best_config.learning_rate:.6f}")
        print(f"  Batch size: {best_config.batch_size}")
        print(f"  Optimizer: {best_config.optimizer}")
        print(f"  Weight decay: {best_config.regularization['weight_decay']:.6f}")
        print(f"  Dropout rate: {best_config.regularization['dropout_rate']:.3f}")
        print(f"  Best score: {results[algorithm]['best_score']:.4f}")
        print(f"  Total trials: {len(history)}")
        print()
    
    # Compare results
    print("--- Comparison ---")
    for algorithm in algorithms:
        result = results[algorithm]
        print(f"{algorithm.upper()}:")
        print(f"  Best score: {result['best_score']:.4f}")
        print(f"  Learning rate: {result['config'].learning_rate:.6f}")
        print(f"  Batch size: {result['config'].batch_size}")
    
    # Plot optimization progress
    try:
        plt.figure(figsize=(12, 5))
        
        for i, algorithm in enumerate(algorithms):
            plt.subplot(1, 2, i + 1)
            history = results[algorithm]['history']
            
            # Extract scores over time
            scores = []
            best_so_far = []
            current_best = 0
            
            for trial in history:
                if trial.metrics:
                    score = trial.metrics.accuracy
                    scores.append(score)
                    current_best = max(current_best, score)
                    best_so_far.append(current_best)
            
            plt.plot(scores, 'o-', alpha=0.7, label='Trial scores')
            plt.plot(best_so_far, 'r-', linewidth=2, label='Best so far')
            plt.xlabel('Trial')
            plt.ylabel('Score')
            plt.title(f'{algorithm.upper()} Optimization Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hyperparameter_optimization_progress.png', dpi=150, bbox_inches='tight')
        print("Optimization progress plot saved as 'hyperparameter_optimization_progress.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping plot generation")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    demonstrate_hyperparameter_optimization()