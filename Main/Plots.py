"""
Plotting Framework for the project





Author: Dylan Myers
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import control
import pathlib
import sys
import os

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Main import cross_dataset_split, d2c
from src.Model_valid_norm import A_matrix
from scripts.Param_Opt import optimal_parameters, construct_TF

# Define sampling time
T = 0.01

# Load data
path = pathlib.Path('/Users/dylanmyers/Desktop/5thyear/MECH412/Project/Main/data/load_data_sc/PRBS_DATA')
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

def simulate_model_on_dataset(tf_continuous, dataset_idx):
    """Simulate the continuous-time transfer function on a dataset"""
    # Extract raw data for this dataset
    data_read = data[dataset_idx, :, :]
    t_raw = data_read[:, 0]
    u_raw = data_read[:, 1]
    y_raw = data_read[:, 2]
    
    # Simulate the transfer function response
    t_sim, y_sim = control.forced_response(tf_continuous, t_raw, u_raw)
    
    return t_raw, u_raw, y_raw, y_sim

def calculate_error_metrics(y_measured, y_simulated):
    """Calculate error metrics between measured and simulated outputs"""
    error = y_measured - y_simulated
    percent_relative_error = (error / np.abs(y_measured)) * 100
    
    # Handle division by zero
    percent_relative_error = np.where(np.abs(y_measured) < 1e-10, 0, percent_relative_error)
    
    return error, percent_relative_error

def plot_optimal_model_all_datasets(order, save_path=None):
    """Single plot showing optimal model on all 4 datasets"""
    # Get optimal model
    analysis = optimal_parameters(order)
    transfer_functions = construct_TF(order, analysis)
    tf_optimal = transfer_functions['optimal']['continuous']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for dataset_idx in range(4):
        t_raw, u_raw, y_raw, y_sim = simulate_model_on_dataset(tf_optimal, dataset_idx)
        error, percent_relative_error = calculate_error_metrics(y_raw, y_sim)
        
        ax = axes[dataset_idx]
        
        # Plot measured vs simulated
        ax.plot(t_raw, y_raw, 'b-', linewidth=2, label='Measured', alpha=0.8)
        ax.plot(t_raw, y_sim, 'r--', linewidth=2, label='Simulated', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Output')
        ax.set_title(f'Dataset {dataset_idx} - Optimal Model (Order {order})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add performance metrics
        mse = np.mean(error**2)
        nmse = mse / np.mean(y_raw**2)
        
        metrics_text = f'NMSE: {nmse:.4f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_error_analysis_single_dataset(tf_continuous, dataset_idx, order, save_path=None):
    """Plot error analysis for a single dataset"""
    t_raw, u_raw, y_raw, y_sim = simulate_model_on_dataset(tf_continuous, dataset_idx)
    error, percent_relative_error = calculate_error_metrics(y_raw, y_sim)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Measured vs Simulated
    ax1.plot(t_raw, y_raw, 'b-', linewidth=2, label='Measured Output', alpha=0.8)
    ax1.plot(t_raw, y_sim, 'r--', linewidth=2, label='Simulated Output', alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Output')
    ax1.set_title(f'Dataset {dataset_idx} - Measured vs Simulated (Order {order})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error vs Time
    ax2.plot(t_raw, error, 'r-', linewidth=1.5, label='Error')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error')
    ax2.set_title(f'Dataset {dataset_idx} - Error vs Time (Order {order})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    nmse = np.mean(error**2) / np.mean(y_raw**2)
    
    stats_text = f'Mean |Error|: {mean_error:.4f}\nMax |Error|: {max_error:.4f}\nNMSE: {nmse:.4f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_optimal_model_error_all_datasets(order, save_path=None):
    """Plot error vs time for optimal model on all 4 datasets"""
    # Get optimal model
    analysis = optimal_parameters(order)
    transfer_functions = construct_TF(order, analysis)
    tf_optimal = transfer_functions['optimal']['continuous']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for dataset_idx in range(4):
        t_raw, u_raw, y_raw, y_sim = simulate_model_on_dataset(tf_optimal, dataset_idx)
        error, percent_relative_error = calculate_error_metrics(y_raw, y_sim)
        
        ax = axes[dataset_idx]
        
        # Plot error vs time
        ax.plot(t_raw, error, 'r-', linewidth=1.5, label='Error')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error')
        ax.set_title(f'Dataset {dataset_idx} - Error vs Time (Optimal Model, Order {order})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add performance metrics
        mean_error = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        nmse = np.mean(error**2) / np.mean(y_raw**2)
        
        metrics_text = f'Mean |Error|: {mean_error:.4f}\nMax |Error|: {max_error:.4f}\nNMSE: {nmse:.4f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_nmse_heatmap(order, save_path=None):
    """Plot NMSE values as a heatmap for easy visualization"""
    from scripts.TF_Val import run_model_comparison
    
    # Get the comparison results
    comparison_results = run_model_comparison(order)
    results_matrix = comparison_results['results_matrix']
    model_names = comparison_results['model_names']
    
    # Extract NMSE values (index 2 in the results matrix)
    nmse_matrix = results_matrix[:, :, 2]  # [model_idx, dataset_idx]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot heatmap
    im = ax.imshow(nmse_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'Dataset {i}' for i in range(4)])
    ax.set_yticks(range(5))
    ax.set_yticklabels(model_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NMSE', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(5):
        for j in range(4):
            text = ax.text(j, i, f'{nmse_matrix[i, j]:.4f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    # Add average column
    avg_values = []
    for i in range(5):
        if i < 4:  # Individual models - exclude training dataset
            test_indices = [k for k in range(4) if k != i]
            avg_nmse = np.mean([nmse_matrix[i, k] for k in test_indices])
        else:  # Optimal model - include all datasets
            avg_nmse = np.mean(nmse_matrix[i, :])
        avg_values.append(avg_nmse)
    
    # Add average column to the right
    for i, avg_val in enumerate(avg_values):
        ax.text(4.5, i, f'{avg_val:.4f}', ha="center", va="center", 
                color="red", fontweight='bold', fontsize=12)
    
    # Add "Average" column header
    ax.text(4.5, -0.5, 'Average', ha="center", va="center", 
            color="red", fontweight='bold', fontsize=12)
    
    ax.set_title(f'NMSE Performance Heatmap (Order {order} System)')
    ax.set_xlabel('Test Dataset')
    ax.set_ylabel('Model')
    
    # Adjust layout to make room for average column
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def generate_simple_plots(order=1, save_plots=True):
    """Generate simplified plots for the assignment"""
    print(f"Generating simplified plots for Order {order} system...")
    
    # Get optimal model
    analysis = optimal_parameters(order)
    transfer_functions = construct_TF(order, analysis)
    tf_optimal = transfer_functions['optimal']['continuous']
    
    plots_generated = []
    
    # 1. Optimal model on all datasets (single plot)
    fig = plot_optimal_model_all_datasets(order)
    if save_plots:
        filename = f'optimal_model_all_datasets_order_{order}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plots_generated.append(filename)
    plt.show()
    
    # 2. Error analysis for one representative dataset
    fig = plot_error_analysis_single_dataset(tf_optimal, 0, order)
    if save_plots:
        filename = f'error_analysis_dataset_0_order_{order}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plots_generated.append(filename)
    plt.show()
    
    # 3. Error vs time for optimal model on all datasets
    fig = plot_optimal_model_error_all_datasets(order)
    if save_plots:
        filename = f'optimal_model_error_all_datasets_order_{order}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plots_generated.append(filename)
    plt.show()
    
    # 4. NMSE heatmap
    fig = plot_nmse_heatmap(order)
    if save_plots:
        filename = f'nmse_heatmap_order_{order}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plots_generated.append(filename)
    plt.show()
    
    if save_plots:
        print(f"\nGenerated {len(plots_generated)} plots:")
        for filename in plots_generated:
            print(f"  - {filename}")
    
    return plots_generated

# %%
if __name__ == "__main__":
    # Generate simplified plots for Order 2 system
    order = 2
    plots = generate_simple_plots(order, save_plots=False)
    
    print(f"\nSimplified plot generation complete for Order {order} system!")
    print("Generated plots address the assignment requirements:")
    print("✓ Optimal model on all datasets")
    print("✓ Error analysis (representative dataset)")
    print("✓ Error vs time for optimal model on all datasets")
    print("✓ NMSE performance heatmap")
    print("✓ Clear dataset labeling")
    print("✓ Performance metrics (NMSE) displayed")
