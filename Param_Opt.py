"""Parameter Optimization and Uncertainty Analysis

This program trains models on each dataset and computes parameter covariance matrices
for uncertainty quantification and statistical validation.

Author: Dylan Myers
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy import linalg
import control
from scipy.linalg import expm, logm
import pathlib


# Import functions from main.py
from Main import cross_dataset_split
from Model_valid_norm import A_matrix
from Main import d2c

# %%
# Load data (reuse from main.py)
path = pathlib.Path('/Users/dylanmyers/Desktop/5thyear/MECH412/Project/load_data_sc/PRBS_DATA')
all_files = sorted(path.glob("*.csv"))
data = [
    np.loadtxt(
        filename,
        dtype=float,
        delimiter=',',
        skiprows=1,
        usecols=(0, 1, 2),
    ) for filename in all_files
]
data = np.array(data)
T = 0.01  # sampling time

# %%
def train_model_on_dataset(dataset_idx, order):
    # Get data for this dataset
    dataset_data = data[dataset_idx, :, :]
    u_raw = dataset_data[:, 1]
    y_raw = dataset_data[:, 2]
    
    # Normalize data
    u_bar = np.max([np.max(u_raw), np.abs(np.min(u_raw))])
    y_bar = np.max([np.max(y_raw), np.abs(np.min(y_raw))])
    uk = u_raw[::-1] / u_bar
    yk = y_raw[::-1] / y_bar
    
    # Build A matrix and solve for parameters
    A, b, _, _ = A_matrix(uk, yk, u_bar, y_bar, order)
    
    # Solve for parameters using least squares
    x = linalg.solve(A.T @ A, A.T @ b)
    
    # Compute parameter covariance matrix
    # Covariance = σ² * (A^T * A)^(-1) where σ² is the residual variance
    residuals = b - A @ x
    sigma_squared = np.mean(residuals**2)
    param_covariance = sigma_squared * linalg.inv(A.T @ A)
    
    # Extract parameter names based on order
    if order == 1:
        param_names = ['a1', 'b1']
    elif order == 2:
        param_names = ['a1', 'a2', 'b1', 'b2']
    elif order == 3:
        param_names = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']
    
    # Create results dictionary
    results = {
        'dataset_idx': dataset_idx,
        'order': order,
        'parameters': x.flatten(),
        'param_names': param_names,
        'covariance_matrix': param_covariance,
        'parameter_std': np.sqrt(np.diag(param_covariance)),
        'residual_variance': sigma_squared,
        'condition_number': np.linalg.cond(A),
        'scaling': {'u_bar': u_bar, 'y_bar': y_bar},
        'n_data_points': len(yk)
    }
    
    return results

# %%
def analyze_all_datasets(order):

    print(f"Training Order {order} models on all datasets...")
    
    all_results = []
    
    # Train on each dataset
    for dataset_idx in range(4):
        print(f"  Training on dataset {dataset_idx}...")
        result = train_model_on_dataset(dataset_idx, order)
        all_results.append(result)
    
    # Analyze parameter consistency across datasets
    param_means = []
    param_stds = []
    
    for i, param_name in enumerate(all_results[0]['param_names']):
        param_values = [result['parameters'][i] for result in all_results]
        param_means.append(np.mean(param_values))
        param_stds.append(np.std(param_values))
    
    # Create summary
    analysis = {
        'order': order,
        'individual_results': all_results,
        'parameter_consistency': {
            'param_names': all_results[0]['param_names'],
            'means_across_datasets': param_means,
            'stds_across_datasets': param_stds,
            'coefficient_of_variation': [std/abs(mean) if mean != 0 else np.inf 
                                       for mean, std in zip(param_means, param_stds)]
        }
    }
    
    return analysis



#%%
def optimal_parameters(order):
    dic_results = analyze_all_datasets(order)   
     
    # Extract parameter estimates and variances for each dataset
    param_array = np.array([res['parameters'] for res in dic_results['individual_results']])  # shape (4, n_params)
    param_names = dic_results['parameter_consistency']['param_names']
    n_params = len(param_names)
    
    # Extract variances from covariance matrices (diagonal elements)
    variance_matrix = np.zeros((4, n_params))
    for i, result in enumerate(dic_results['individual_results']):
        variance_matrix[i, :] = np.diag(result['covariance_matrix'])
    
    # UNNORMALIZE parameters and variances to common scale before optimal weighting
    unnormalized_params = np.zeros((4, n_params))
    unnormalized_variances = np.zeros((4, n_params))
    
    for dataset_idx in range(4):
        result = dic_results['individual_results'][dataset_idx]
        u_bar = result['scaling']['u_bar']
        y_bar = result['scaling']['y_bar']
        parameters = result['parameters']
        cov_matrix = result['covariance_matrix']
        
        # Unnormalize parameters and variances
        scaling_factor = y_bar / u_bar
        
        # a parameters (denominator coefficients) stay the same
        for i in range(order):
            unnormalized_params[dataset_idx, i] = parameters[i]
            unnormalized_variances[dataset_idx, i] = cov_matrix[i, i]
        
        # b parameters (numerator coefficients) get scaled
        for i in range(order, n_params):
            unnormalized_params[dataset_idx, i] = parameters[i] * scaling_factor
            unnormalized_variances[dataset_idx, i] = cov_matrix[i, i] * scaling_factor**2
    
    # Calculate optimal parameters using inverse-variance weighting on UNNORMALIZED parameters
    optimal_params = np.zeros(n_params)
    optimal_variances = np.zeros(n_params)
    normalized_weights_matrix = np.zeros((4, n_params))
    
    #refactored into a loop in order to work for any inputted order
    #loops over each parameter, includes all 4 of the covariance matrices though.
    for param_idx in range(n_params):
        # Use UNNORMALIZED parameters and variances
        param_estimates = unnormalized_params[:, param_idx]
        param_variances = unnormalized_variances[:, param_idx]
        raw_weights = 1.0 / param_variances
        normalized_weights = raw_weights / np.sum(raw_weights)
        normalized_weights_matrix[:, param_idx] = normalized_weights
        # Calculate optimal parameter and variance
        optimal_params[param_idx] = np.sum(param_estimates * normalized_weights)
        optimal_variances[param_idx] = 1.0 / np.sum(raw_weights)
    
    # Create results dictionary
    results = {
        'order': order,
        'parameter_names': param_names,
        'optimal_parameters': optimal_params,

        #not needed as of right now
        'optimal_variances': optimal_variances,
        'normalized_weights': normalized_weights_matrix,  # weights that sum to 1
        'parameter_consistency': dic_results['parameter_consistency']}
    
    return results

def construct_TF(order, optimal_params):
    """
    Construct 5 transfer functions:
    """
    # Get individual results for each dataset
    dic_results = analyze_all_datasets(order)
    
    # Initialize transfer functions dictionary
    transfer_functions = {}
    
    # Construct individual TFs for each dataset
    for i, result in enumerate(dic_results['individual_results']):
        dataset_idx = result['dataset_idx']
        parameters = result['parameters']
        u_bar = result['scaling']['u_bar']
        y_bar = result['scaling']['y_bar']
        
        # Extract TF coefficients (generic for any order)
        den_coeffs = np.hstack([1, parameters[:order]])  # [1, a1, a2, a3, ...]
        num_coeffs = parameters[order:]  # [b1, b2, b3, ...]
        
        # Create discrete-time TF (UNNORMALIZED)
        Pd = y_bar / u_bar * control.tf(num_coeffs, den_coeffs, T)
        
        # Convert to continuous-time
        Pc = d2c(Pd)
        
        transfer_functions[f'dataset_{dataset_idx}'] = {
            'discrete': Pd,
            'continuous': Pc,
            'parameters': parameters,
            'scaling': {'u_bar': u_bar, 'y_bar': y_bar}
        }
    
    # Construct optimal TF
    optimal_parameters = optimal_params['optimal_parameters']
    

    # Extract TF coefficients for optimal parameters (generic for any order)
    den_coeffs = np.hstack([1, optimal_parameters[:order]])  # [1, a1, a2, a3, ...]
    num_coeffs = optimal_parameters[order:]  # [b1, b2, b3, ...]
    
    # No need to apply (y_bar/u_bar) scaling since optimal parameters are in physical units
    Pd_optimal = control.tf(num_coeffs, den_coeffs, T)
    Pc_optimal = d2c(Pd_optimal)
    
    transfer_functions['optimal'] = {
        'discrete': Pd_optimal,
        'continuous': Pc_optimal,
        'parameters': optimal_parameters,
    }
    return transfer_functions




# %%
# Example usage
if __name__ == "__main__":
    # Analyze Order 2 system
    order = 2
    analysis = optimal_parameters(order)
    
    # Print optimal results table
    '''
    analysis = optimal_parameters(order)

    print(f"\n{'='*30}")
    print(f"OPTIMAL PARAMETERS - ORDER {order} SYSTEM")
    print(f"{'Parameter':<12} {'Optimal Value':<15}")
    
    # Print each parameter's optimal results
    for i, param_name in enumerate(analysis['parameter_names']):
        optimal_val = analysis['optimal_parameters'][i]
        print(f"{param_name:<12} {optimal_val:<15.8f}")
    
    print("-" * 30)
    '''
    
    transfer_functions = construct_TF(order, analysis)


    
    # Print TF information
    for tf_name, tf_data in transfer_functions.items():
        print(f"{'='*50}")
        print(f"\n{tf_name.upper()}:")
        print(f"  Discrete: {tf_data['discrete']}")
        print(f"  Continuous: {tf_data['continuous']}")
        print(f"{'='*50}")
    
    print(f"\n{'='*50}")
    print("TRANSFER FUNCTIONS CONSTRUCTED SUCCESSFULLY!")
    print(f"{'='*50}")

