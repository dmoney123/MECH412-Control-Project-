"""
Transfer Function Validation

This program validates the optimal transfer function against all datasets
by calculating unnormalized NMSE, MSE, and MSO metrics.

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

#simulate data with CT TFs and control.forced_response
def calculate_metrics_on_dataset(tf_continuous, dataset_idx, order):
    # Extract raw data for this dataset
    data_read = data[dataset_idx, :, :]
    t_raw = data_read[:, 0]
    u_raw = data_read[:, 1]
    y_raw = data_read[:, 2]
    
    # Simulate the transfer function response
    t_sim, y_sim = control.forced_response(tf_continuous, t_raw, u_raw)
    
    # Calculate metrics (unnormalized)
    mse = np.mean((y_raw - y_sim)**2)
    mso = np.mean(y_raw**2)
    nmse = mse / mso
    
    # Calculate %VAF (Variance Accounted For)
    # %VAF = 100 * (1 - var(y_actual - y_simulated) / var(y_actual))
    var_actual = np.var(y_raw)
    var_error = np.var(y_raw - y_sim)
    vaf_percent = 100 * (1 - var_error / var_actual) if var_actual > 0 else 0
    
    # Calculate fit ratio
    # Fit ratio = std(y_simulated) / std(y_actual)
    std_actual = np.std(y_raw)
    std_sim = np.std(y_sim)
    fit_ratio = std_sim / std_actual if std_actual > 0 else 0
    
    return {
        'dataset_idx': dataset_idx,
        'mse': mse,
        'mso': mso,
        'nmse': nmse,
        'vaf_percent': vaf_percent,
        'fit_ratio': fit_ratio,
        'n_points': len(y_raw)
    }



def test_individual_model(model_idx, transfer_functions, results_matrix, model_names, order):
    """Test individual dataset model on all datasets"""
    model_name = f'dataset_{model_idx}'
    tf_model = transfer_functions[model_name]['continuous']
    u_bar = transfer_functions[model_name]['scaling']['u_bar']
    y_bar = transfer_functions[model_name]['scaling']['y_bar']
    
    print(f"\n{model_names[model_idx]}:")
    print(f"  Trained on Dataset {model_idx} (u_bar={u_bar:.2f}, y_bar={y_bar:.2f})")
    
    for test_dataset_idx in range(4):
        metrics = calculate_metrics_on_dataset(tf_model, test_dataset_idx, order)
        results_matrix[model_idx, test_dataset_idx, :] = [metrics['mse'], metrics['mso'], metrics['nmse'], metrics['vaf_percent'], metrics['fit_ratio']]
        print(f"    Test on Dataset {test_dataset_idx}: NMSE = {metrics['nmse']:.6f}")

def test_optimal_model(transfer_functions, results_matrix, model_names, order):
    """Test optimal model on all datasets"""
    tf_optimal = transfer_functions['optimal']['continuous']
    print(f"\n{model_names[4]}:")
    print(f"  Inverse-variance weighted optimal model")
    
    for test_dataset_idx in range(4):
        metrics = calculate_metrics_on_dataset(tf_optimal, test_dataset_idx, order)
        results_matrix[4, test_dataset_idx, :] = [metrics['mse'], metrics['mso'], metrics['nmse'], metrics['vaf_percent'], metrics['fit_ratio']]
        print(f"    Test on Dataset {test_dataset_idx}: NMSE = {metrics['nmse']:.6f}")

def print_summary_table(results_matrix, model_names):
    """Print comprehensive summary tables for all metrics"""
    
    # NMSE Summary Table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - NMSE VALUES")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Dataset 0':<12} {'Dataset 1':<12} {'Dataset 2':<12} {'Dataset 3':<12} {'Average':<12}")
    print("-" * 80)
    
    for model_idx in range(5):
        nmse_values = results_matrix[model_idx, :, 2]
        
        # Calculate average (exclude training dataset for individual models)
        if model_idx < 4:
            test_indices = [i for i in range(4) if i != model_idx]
            avg_nmse = np.mean([nmse_values[i] for i in test_indices])
        else:
            avg_nmse = np.mean(nmse_values)
        
        print(f"{model_names[model_idx]:<20} {nmse_values[0]:<12.6f} {nmse_values[1]:<12.6f} {nmse_values[2]:<12.6f} {nmse_values[3]:<12.6f} {avg_nmse:<12.6f}")
    
    # %VAF Summary Table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - %VAF VALUES")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Dataset 0':<12} {'Dataset 1':<12} {'Dataset 2':<12} {'Dataset 3':<12} {'Average':<12}")
    print("-" * 80)
    
    for model_idx in range(5):
        vaf_values = results_matrix[model_idx, :, 3]
        
        # Calculate average (exclude training dataset for individual models)
        if model_idx < 4:
            test_indices = [i for i in range(4) if i != model_idx]
            avg_vaf = np.mean([vaf_values[i] for i in test_indices])
        else:
            avg_vaf = np.mean(vaf_values)
        
        print(f"{model_names[model_idx]:<20} {vaf_values[0]:<12.2f} {vaf_values[1]:<12.2f} {vaf_values[2]:<12.2f} {vaf_values[3]:<12.2f} {avg_vaf:<12.2f}")
    
    # Fit Ratio Summary Table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - FIT RATIO VALUES")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Dataset 0':<12} {'Dataset 1':<12} {'Dataset 2':<12} {'Dataset 3':<12} {'Average':<12}")
    print("-" * 80)
    
    for model_idx in range(5):
        fit_ratio_values = results_matrix[model_idx, :, 4]
        
        # Calculate average (exclude training dataset for individual models)
        if model_idx < 4:
            test_indices = [i for i in range(4) if i != model_idx]
            avg_fit_ratio = np.mean([fit_ratio_values[i] for i in test_indices])
        else:
            avg_fit_ratio = np.mean(fit_ratio_values)
        
        print(f"{model_names[model_idx]:<20} {fit_ratio_values[0]:<12.4f} {fit_ratio_values[1]:<12.4f} {fit_ratio_values[2]:<12.4f} {fit_ratio_values[3]:<12.4f} {avg_fit_ratio:<12.4f}")

def print_best_models_per_dataset(results_matrix, model_names):
    """Print best model for each dataset"""
    print(f"\n{'='*80}")
    print("BEST MODEL FOR EACH DATASET:")
    print(f"{'='*80}")
    
    for test_dataset_idx in range(4):
        nmse_values = results_matrix[:, test_dataset_idx, 2]
        best_model_idx = np.argmin(nmse_values)
        best_nmse = nmse_values[best_model_idx]
        print(f"Dataset {test_dataset_idx}: {model_names[best_model_idx]} (NMSE = {best_nmse:.6f})")

def calculate_rankings(results_matrix, model_names):
    """Calculate and print overall model rankings for all metrics"""
    
    # Calculate average metrics for each model
    avg_metrics_per_model = np.zeros((5, 3))  # [model_idx, metric_idx] where metric_idx: 0=NMSE, 1=%VAF, 2=Fit Ratio
    
    for model_idx in range(5):
        if model_idx < 4:
            test_indices = [i for i in range(4) if i != model_idx]
        else:
            test_indices = list(range(4))
        
        # NMSE (lower is better)
        nmse_values = results_matrix[model_idx, test_indices, 2]
        avg_metrics_per_model[model_idx, 0] = np.mean(nmse_values)
        
        # %VAF (higher is better)
        vaf_values = results_matrix[model_idx, test_indices, 3]
        avg_metrics_per_model[model_idx, 1] = np.mean(vaf_values)
        
        # Fit Ratio (closer to 1.0 is better)
        fit_ratio_values = results_matrix[model_idx, test_indices, 4]
        avg_metrics_per_model[model_idx, 2] = np.mean(fit_ratio_values)
    
    # NMSE Rankings (lower is better)
    nmse_ranking_indices = np.argsort(avg_metrics_per_model[:, 0])
    print(f"\n{'='*80}")
    print("OVERALL MODEL RANKING (by average NMSE - lower is better):")
    print(f"{'='*80}")
    for rank, model_idx in enumerate(nmse_ranking_indices):
        print(f"{rank + 1}. {model_names[model_idx]}: {avg_metrics_per_model[model_idx, 0]:.6f}")
    
    # %VAF Rankings (higher is better)
    vaf_ranking_indices = np.argsort(avg_metrics_per_model[:, 1])[::-1]  # Reverse for descending order
    print(f"\n{'='*80}")
    print("OVERALL MODEL RANKING (by average %VAF - higher is better):")
    print(f"{'='*80}")
    for rank, model_idx in enumerate(vaf_ranking_indices):
        print(f"{rank + 1}. {model_names[model_idx]}: {avg_metrics_per_model[model_idx, 1]:.2f}%")
    
    # Fit Ratio Rankings (closer to 1.0 is better)
    fit_ratio_ranking_indices = np.argsort(np.abs(avg_metrics_per_model[:, 2] - 1.0))
    print(f"\n{'='*80}")
    print("OVERALL MODEL RANKING (by average Fit Ratio - closer to 1.0 is better):")
    print(f"{'='*80}")
    for rank, model_idx in enumerate(fit_ratio_ranking_indices):
        print(f"{rank + 1}. {model_names[model_idx]}: {avg_metrics_per_model[model_idx, 2]:.4f}")
    
    return nmse_ranking_indices, avg_metrics_per_model

def calculate_optimal_model_uncertainty(order):
    """
    Simple function to calculate parameter uncertainty for the optimal model.
    Based on the approach from 0_sys_ID_sc.py
    """
    from Param_Opt import analyze_all_datasets, optimal_parameters
    
    # Get optimal parameters
    analysis = optimal_parameters(order)
    optimal_params = analysis['optimal_parameters']
    
    # Get individual model results for uncertainty calculation
    individual_analysis = analyze_all_datasets(order)
    
    # Calculate average test error across all datasets for sigma
    total_test_errors = []
    total_test_points = 0
    
    for dataset_idx in range(4):
        result = individual_analysis['individual_results'][dataset_idx]
        params = result['parameters']
        cov_matrix = result['covariance_matrix']
        
        # Get test data (other 3 datasets)
        test_indices = [i for i in range(4) if i != dataset_idx]
        
        for test_idx in test_indices:
            # Load test data
            data_read = data[test_idx, :, :]
            t_raw = data_read[:, 0]
            u_raw = data_read[:, 1]
            y_raw = data_read[:, 2]
            
            # Build A matrix for this test dataset
            A_test, b_test, _, _ = A_matrix(u_raw, y_raw, 0, 0, order)
            
            # Calculate prediction errors
            y_pred = A_test @ params
            errors = b_test.flatten() - y_pred.flatten()
            
            total_test_errors.extend(errors)
            total_test_points += len(errors)
    
    # Calculate overall test standard deviation
    sigma_test = np.sqrt(np.mean(np.array(total_test_errors)**2))
    
    # Calculate parameter uncertainties using inverse-variance weighting
    # Get individual parameter variances
    individual_variances = []
    for dataset_idx in range(4):
        result = individual_analysis['individual_results'][dataset_idx]
        cov_matrix = result['covariance_matrix']
        individual_variances.append(np.diag(cov_matrix))
    
    individual_variances = np.array(individual_variances)
    
    # Calculate optimal parameter uncertainties using inverse-variance weighting
    optimal_variances = np.zeros(len(optimal_params))
    for param_idx in range(len(optimal_params)):
        individual_vars = individual_variances[:, param_idx]
        # Avoid division by zero
        individual_vars = np.maximum(individual_vars, 1e-12)
        optimal_variances[param_idx] = 1.0 / np.sum(1.0 / individual_vars)
    
    # Calculate parameter standard deviations and relative uncertainties
    param_std = np.sqrt(optimal_variances)
    param_rel_unc = np.abs(param_std / optimal_params)
    
    # Generate parameter names
    param_names = []
    for i in range(order):
        param_names.append(f'a{i+1}')
    for i in range(order):
        param_names.append(f'b{i+1}')
    
    return {
        'optimal_parameters': optimal_params,
        'parameter_std': param_std,
        'parameter_rel_unc': param_rel_unc,
        'parameter_names': param_names,
        'test_sigma': sigma_test
    }

def print_optimal_uncertainty(uncertainty_results, order):
    """Print optimal model uncertainty in a simple format"""
    print(f"\n{'='*60}")
    print(f"OPTIMAL MODEL PARAMETER UNCERTAINTY - ORDER {order} SYSTEM")
    print(f"{'='*60}")
    print(f"Test Standard Deviation: {uncertainty_results['test_sigma']:.6f}")
    print()
    print("PARAMETER UNCERTAINTIES:")
    print("-" * 50)
    
    for i, (param_name, param_value, param_std, param_rel_unc) in enumerate(zip(
        uncertainty_results['parameter_names'],
        uncertainty_results['optimal_parameters'],
        uncertainty_results['parameter_std'],
        uncertainty_results['parameter_rel_unc']
    )):
        print(f"{param_name}: {param_value:.6f} ± {param_std:.6f} ({param_rel_unc*100:.2f}%)")

def run_model_comparison(order):
    # Get optimal parameters and construct all TFs
    analysis = optimal_parameters(order)
    transfer_functions = construct_TF(order, analysis)
    
    # Initialize results matrix: [model_idx, dataset_idx, metric_idx]
    # metric_idx: 0=MSE, 1=MSO, 2=NMSE, 3=%VAF, 4=Fit Ratio
    results_matrix = np.zeros((5, 4, 5))
    
    # Model names
    model_names = ['Dataset 0 Model', 'Dataset 1 Model', 'Dataset 2 Model', 'Dataset 3 Model', 'Optimal Model']
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE MODEL COMPARISON - ORDER {order} SYSTEM")
    print(f"{'='*80}")
    
    # Test individual models
    for model_idx in range(4):
        test_individual_model(model_idx, transfer_functions, results_matrix, model_names, order)
    
    # Test optimal model
    test_optimal_model(transfer_functions, results_matrix, model_names, order)
    
    # Print summary tables
    print_summary_table(results_matrix, model_names)
    
    # Print best models per dataset
    print_best_models_per_dataset(results_matrix, model_names)
    
    # Calculate and print rankings
    ranking_indices, avg_metrics_per_model = calculate_rankings(results_matrix, model_names)
    
    return {
        'results_matrix': results_matrix,
        'model_names': model_names,
        'order': order,
        'ranking': ranking_indices,
        'avg_metrics_per_model': avg_metrics_per_model
    }


# %%
# Example usage
if __name__ == "__main__":
    # Run model comparison for Order 1
    order = 1
    comparison_results = run_model_comparison(order)
    
    print(f"\nComparison complete!")
    print(f"Best overall model: {comparison_results['model_names'][comparison_results['ranking'][0]]}")
    print(f"  Average NMSE: {comparison_results['avg_metrics_per_model'][comparison_results['ranking'][0], 0]:.6f}")
    print(f"  Average %VAF: {comparison_results['avg_metrics_per_model'][comparison_results['ranking'][0], 1]:.2f}%")
    print(f"  Average Fit Ratio: {comparison_results['avg_metrics_per_model'][comparison_results['ranking'][0], 2]:.4f}")
    
    # Calculate and print optimal model uncertainty
    print(f"\n{'='*60}")
    print("CALCULATING OPTIMAL MODEL UNCERTAINTY...")
    print(f"{'='*60}")
    
    uncertainty_results = calculate_optimal_model_uncertainty(order)
    print_optimal_uncertainty(uncertainty_results, order)
    