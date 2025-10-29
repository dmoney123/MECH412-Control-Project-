"""
Theoretical Transfer Function Validation

This program validates models using theoretical NMSE calculations
directly from the least squares solution, without transfer function conversion.

Author: Dylan Myers
Date: 2025
"""

import numpy as np
import control
import pathlib
from Main import cross_dataset_split, d2c
from Model_valid_norm import A_matrix
from Param_Opt import optimal_parameters, construct_TF

# Define sampling time
T = 0.01

# %%
# Load data
path = pathlib.Path('/Users/dylanmyers/Desktop/5thyear/MECH412/Project/load_data_sc/PRBS_DATA')
all_files = sorted(path.glob("*.csv"))

# Load all datasets
data = np.zeros((4, 1000, 3))
for i, file in enumerate(all_files):
    data[i, :, :] = np.loadtxt(file, delimiter=',')

# %%
def calculate_theoretical_metrics_on_dataset(model_parameters, model_scaling, dataset_idx, order):
    """
    Calculate theoretical NMSE using unnormalized parameters on raw data
    
    Parameters:
    -----------
    model_parameters : array
        Model parameters from least squares solution (normalized)
    model_scaling : dict
        Scaling factors {'u_bar': u_bar, 'y_bar': y_bar} from training dataset
    dataset_idx : int
        Test dataset index (0, 1, 2, or 3)
    order : int
        Model order
    
    Returns:
    --------
    dict with theoretical metrics
    """
    # Extract raw data for test dataset
    data_read = data[dataset_idx, :, :]
    t_raw = data_read[:, 0]
    u_raw = data_read[:, 1]
    y_raw = data_read[:, 2]
    
    # Unnormalize parameters to physical units
    u_bar = model_scaling['u_bar']
    y_bar = model_scaling['y_bar']
    scaling_factor = y_bar / u_bar
    
    # Unnormalize parameters (same logic as in Param_Opt.py)
    unnormalized_params = model_parameters.copy()
    for i in range(order, len(model_parameters)):  # b parameters get scaled
        unnormalized_params[i] = model_parameters[i] * scaling_factor
    
    # CRITICAL: Use flipped data to match A matrix construction
    # The A matrix was designed for flipped data, so we need to flip here too
    u_flipped = u_raw[::-1]
    y_flipped = y_raw[::-1]
    
    # Build A matrix and b vector using flipped data (to match training)
    A_raw, b_raw, _, _ = A_matrix(u_flipped, y_flipped, 1.0, 1.0, order)  # No scaling
    
    # Calculate theoretical metrics using unnormalized parameters
    # The key insight: b_raw represents y[k] values, and we predict them using A*x
    # This should give the same result as TF simulation, but let's verify
    
    residuals = b_raw - A_raw @ unnormalized_params
    mse_theo = np.mean(residuals**2)
    mso_theo = np.mean(b_raw**2)
    nmse_theo = mse_theo / mso_theo if mso_theo > 0 else float('inf')
    
    return {
        'dataset_idx': dataset_idx,
        'mse': mse_theo,
        'mso': mso_theo,
        'nmse': nmse_theo,
        'n_points': len(b_raw)
    }

def calculate_theoretical_metrics_optimal(optimal_parameters_unnorm, dataset_idx, order):
    """
    Calculate theoretical metrics for optimal model using unnormalized parameters
    
    Parameters:
    -----------
    optimal_parameters_unnorm : array
        Unnormalized optimal parameters (in physical units)
    dataset_idx : int
        Test dataset index (0, 1, 2, or 3)
    order : int
        Model order
    
    Returns:
    --------
    dict with theoretical metrics
    """
    # Extract raw data for test dataset
    data_read = data[dataset_idx, :, :]
    t_raw = data_read[:, 0]
    u_raw = data_read[:, 1]
    y_raw = data_read[:, 2]
    
    # CRITICAL: Use flipped data to match A matrix construction
    # The A matrix was designed for flipped data, so we need to flip here too
    u_flipped = u_raw[::-1]
    y_flipped = y_raw[::-1]
    
    # Build A matrix and b vector using flipped data (to match training)
    A_raw, b_raw, _, _ = A_matrix(u_flipped, y_flipped, 1.0, 1.0, order)  # No scaling
    
    # Calculate theoretical metrics using unnormalized parameters
    # NMSE = ||b_raw - A_raw * x_unnorm||²₂ / ||b_raw||²₂
    residuals = b_raw - A_raw @ optimal_parameters_unnorm
    mse_theo = np.mean(residuals**2)
    mso_theo = np.mean(b_raw**2)
    nmse_theo = mse_theo / mso_theo if mso_theo > 0 else float('inf')
    
    return {
        'dataset_idx': dataset_idx,
        'mse': mse_theo,
        'mso': mso_theo,
        'nmse': nmse_theo,
        'n_points': len(b_raw)
    }

def compare_all_models_theoretical(order):
    """
    Compare all 5 models using theoretical NMSE calculations
    
    Parameters:
    -----------
    order : int
        Model order (1, 2, or 3)
    
    Returns:
    --------
    dict with comparison results
    """
    # Get optimal parameters and construct all TFs
    analysis = optimal_parameters(order)
    transfer_functions = construct_TF(order, analysis)
    
    # Initialize results matrix: [model_idx, test_dataset_idx]
    results_matrix = np.zeros((5, 4, 3))  # 5 models, 4 test datasets, 3 metrics (MSE, MSO, NMSE)
    model_names = ['Dataset 0 Model', 'Dataset 1 Model', 'Dataset 2 Model', 'Dataset 3 Model', 'Optimal Model']
    
    print(f"\n{'='*80}")
    print(f"THEORETICAL MODEL COMPARISON - ORDER {order} SYSTEM")
    print(f"{'='*80}")
    
    # Test each model on each dataset
    for model_idx in range(5):
        if model_idx < 4:  # Individual dataset models
            model_name = f'dataset_{model_idx}'
            model_parameters = transfer_functions[model_name]['parameters']
            model_scaling = transfer_functions[model_name]['scaling']
            
            print(f"\n{model_names[model_idx]}:")
            print(f"  Trained on Dataset {model_idx} (u_bar={model_scaling['u_bar']:.2f}, y_bar={model_scaling['y_bar']:.2f})")
            
            for test_dataset_idx in range(4):
                # Calculate theoretical metrics
                metrics = calculate_theoretical_metrics_on_dataset(
                    model_parameters, model_scaling, test_dataset_idx, order)
                
                results_matrix[model_idx, test_dataset_idx, 0] = metrics['mse']
                results_matrix[model_idx, test_dataset_idx, 1] = metrics['mso']
                results_matrix[model_idx, test_dataset_idx, 2] = metrics['nmse']
                
                print(f"    Test on Dataset {test_dataset_idx}: NMSE = {metrics['nmse']:.6f}")
        
        else:  # Optimal model
            # For optimal model, we need to work with unnormalized parameters
            # We'll calculate theoretical metrics in physical space for each dataset
            optimal_parameters_unnorm = analysis['optimal_parameters']
            
            print(f"\n{model_names[model_idx]}:")
            print(f"  Inverse-variance weighted optimal model (unnormalized)")
            
            for test_dataset_idx in range(4):
                # Calculate theoretical metrics for optimal model in physical space
                metrics = calculate_theoretical_metrics_optimal(
                    optimal_parameters_unnorm, test_dataset_idx, order)
                
                results_matrix[model_idx, test_dataset_idx, 0] = metrics['mse']
                results_matrix[model_idx, test_dataset_idx, 1] = metrics['mso']
                results_matrix[model_idx, test_dataset_idx, 2] = metrics['nmse']
                
                print(f"    Test on Dataset {test_dataset_idx}: NMSE = {metrics['nmse']:.6f}")
    
    # Calculate summary statistics
    print(f"\n{'='*80}")
    print("THEORETICAL SUMMARY TABLE - NMSE VALUES")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Dataset 0':<12} {'Dataset 1':<12} {'Dataset 2':<12} {'Dataset 3':<12} {'Average':<12}")
    print("-" * 80)
    
    for model_idx in range(5):
        nmse_values = results_matrix[model_idx, :, 2]  # NMSE values
        
        if model_idx < 4:  # Individual dataset models - exclude training dataset
            # Create mask to exclude the training dataset
            test_indices = [i for i in range(4) if i != model_idx]
            avg_nmse = np.mean([nmse_values[i] for i in test_indices])
        else:  # Optimal model - include all datasets
            avg_nmse = np.mean(nmse_values)
        
        print(f"{model_names[model_idx]:<20} {nmse_values[0]:<12.6f} {nmse_values[1]:<12.6f} {nmse_values[2]:<12.6f} {nmse_values[3]:<12.6f} {avg_nmse:<12.6f}")
    
    # Find best model for each dataset
    print(f"\n{'='*80}")
    print("BEST THEORETICAL MODEL FOR EACH DATASET:")
    print(f"{'='*80}")
    
    for test_dataset_idx in range(4):
        nmse_values = results_matrix[:, test_dataset_idx, 2]
        best_model_idx = np.argmin(nmse_values)
        best_nmse = nmse_values[best_model_idx]
        print(f"Dataset {test_dataset_idx}: {model_names[best_model_idx]} (NMSE = {best_nmse:.6f})")
    
    # Calculate overall rankings (excluding training datasets for individual models)
    avg_nmse_per_model = np.zeros(5)
    for model_idx in range(5):
        nmse_values = results_matrix[model_idx, :, 2]
        if model_idx < 4:  # Individual dataset models - exclude training dataset
            test_indices = [i for i in range(4) if i != model_idx]
            avg_nmse_per_model[model_idx] = np.mean([nmse_values[i] for i in test_indices])
        else:  # Optimal model - include all datasets
            avg_nmse_per_model[model_idx] = np.mean(nmse_values)
    
    ranking_indices = np.argsort(avg_nmse_per_model)
    
    print(f"\n{'='*80}")
    print("OVERALL THEORETICAL MODEL RANKING (by average NMSE):")
    print(f"{'='*80}")
    
    for rank, model_idx in enumerate(ranking_indices):
        print(f"{rank + 1}. {model_names[model_idx]}: {avg_nmse_per_model[model_idx]:.6f}")
    
    return {
        'results_matrix': results_matrix,
        'model_names': model_names,
        'order': order,
        'ranking': ranking_indices,
        'avg_nmse_per_model': avg_nmse_per_model
    }

# %%
# Example usage
if __name__ == "__main__":
    # Compare all models using theoretical calculations for Order 1
    order = 1
    comparison_results = compare_all_models_theoretical(order)
    
    print(f"\nTheoretical comparison complete!")
    print(f"Best theoretical model: {comparison_results['model_names'][comparison_results['ranking'][0]]}")
    print(f"  Average NMSE: {comparison_results['avg_nmse_per_model'][comparison_results['ranking'][0]]:.6f}")
