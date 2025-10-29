"""


This code performs cross-dataset validation of the model order.

It works by training a three orders (1.2.3) of models on one dataset and then testing it on the other three.
The trained dataset is then rotated and another 3 models are trained and tested on the new dataset.
This process is repeated for all four datasets.


The goal is to investigate which model order we should use to model our system

As of right now, this isn't explicitly used in the project, but this is still useful for future reference.



Author: Dylan Myers
References: J R Forbes, MECH 412 Lecture Notes
Date: October 2025
"""

# %%
# Libraries
from pickle import NONE
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import pathlib
from scipy import linalg
import control
from scipy.linalg import expm, logm

#Credit to J R Forbes for the d2c function#

# %%
# Read in all input-output (IO) data
path = pathlib.Path('/Users/dylanmyers/Desktop/5thyear/MECH412/Project/Main/data/load_data_sc/PRBS_DATA')
all_files = sorted(path.glob("*.csv"))
# all_files.sort()
data = [
    np.loadtxt(
        filename,
        dtype=float,
        delimiter=',',
        skiprows=1,
        usecols=(0, 1, 2),
    ) for filename in all_files
]
#3d array with 4 files (four i/o data sets)#
data = np.array(data)

# Define sampling time (assuming all datasets have same sampling time)
T = 0.01  # seconds - adjust this to match your actual sampling time

# %%
# This split chooses one of the datasets as training, then allows us to test on the other three
def cross_dataset_split(train_dataset=0):
    # Load training dataset
    train_data = data[train_dataset, :, :]
    t_train = train_data[:, 0]
    u_train = train_data[:, 1]  # V, volts
    y_train = train_data[:, 2]  # LPM, liters per minute
    
    # Normalize using training data only
    u_bar = np.max([np.max(u_train), np.abs(np.min(u_train))])
    y_bar = np.max([np.max(y_train), np.abs(np.min(y_train))])
    
    # Normalize training data
    uk_train = u_train[::-1] / u_bar
    yk_train = y_train[::-1] / y_bar
    
    # Load and normalize test datasets
    test_datasets = {}
    for i in range(4):
        if i != train_dataset:
            test_data = data[i, :, :]
            t_test = test_data[:, 0]
            u_test = test_data[:, 1]
            y_test = test_data[:, 2]
            
            # Normalize using same scaling factors as training
            uk_test = u_test[::-1] / u_bar
            yk_test = y_test[::-1] / y_bar
            
            test_datasets[i] = {
                'time': t_test,
                'input': uk_test,
                'output': yk_test,
                'input_raw': u_test,
                'output_raw': y_test
            }
    
    return {
        'train': {
            'time': t_train,
            'input': uk_train,
            'output': yk_train,
            'input_raw': u_train,
            'output_raw': y_train
        },
        'test': test_datasets,
        'scaling': {'u_bar': u_bar, 'y_bar': y_bar}
    }

# Cross-dataset split: train on dataset 0, test on datasets 1, 2, 3
cross_dataset_data = cross_dataset_split(train_dataset=2)
uk = cross_dataset_data['train']['input']
yk = cross_dataset_data['train']['output']
u_bar = cross_dataset_data['scaling']['u_bar']
y_bar = cross_dataset_data['scaling']['y_bar']

#Inspired by lecture notes 7a-sysID# --> CONSIDER REFACTORING <---
def A_matrix(uk, yk, u_bar, y_bar, order):
    N_train = len(yk)
    
    if order == 1:
        # 1st Order: y[k] = -a1*y[k-1] + b1*u[k-1]
        b = yk[:-1].reshape(-1, 1)
        A = np.zeros((N_train - 1, 2))
        A[:, [0]] = -yk[1:].reshape(-1, 1) 
        A[:, [1]] = uk[1:].reshape(-1, 1)   
        
    elif order == 2:
        # 2nd Order: y[k] = -a1*y[k-1] - a2*y[k-2] + b1*u[k-1] + b2*u[k-2]
        b = yk[:-2].reshape(-1, 1)
        A = np.zeros((N_train - 2, 4))
        A[:, [0]] = -yk[1:-1].reshape(-1, 1)  # -a1*y[k-1]
        A[:, [1]] = -yk[2:].reshape(-1, 1)    # -a2*y[k-2]
        A[:, [2]] = uk[1:-1].reshape(-1, 1)   # b1*u[k-1]
        A[:, [3]] = uk[2:].reshape(-1, 1)     # b2*u[k-2]
        
    elif order == 3:
        # 3rd Order: y[k] = -a1*y[k-1] - a2*y[k-2] - a3*y[k-3] + b1*u[k-1] + b2*u[k-2] + b3*u[k-3]
        b = yk[:-3].reshape(-1, 1)
        A = np.zeros((N_train - 3, 6))
        A[:, [0]] = -yk[1:-2].reshape(-1, 1)  # -a1*y[k-1]
        A[:, [1]] = -yk[2:-1].reshape(-1, 1)  # -a2*y[k-2]
        A[:, [2]] = -yk[3:].reshape(-1, 1)    # -a3*y[k-3]
        A[:, [3]] = uk[1:-2].reshape(-1, 1)   # b1*u[k-1]
        A[:, [4]] = uk[2:-1].reshape(-1, 1)   # b2*u[k-2]
        A[:, [5]] = uk[3:].reshape(-1, 1)     # b3*u[k-3]

    elif order == 4:
        # 4th Order: y[k] = -a1*y[k-1] - a2*y[k-2] - a3*y[k-3] - a4*y[k-4] + b1*u[k-1] + b2*u[k-2] + b3*u[k-3] + b4*u[k-4]
        b = yk[:-4].reshape(-1, 1)
        A = np.zeros((N_train - 4, 8))
        A[:, [0]] = -yk[1:-3].reshape(-1, 1)  # -a1*y[k-1]
        A[:, [1]] = -yk[2:-2].reshape(-1, 1)  # -a2*y[k-2]
        A[:, [2]] = -yk[3:-1].reshape(-1, 1)  # -a3*y[k-3]
        A[:, [3]] = -yk[4:].reshape(-1, 1)    # -a4*y[k-4]
        A[:, [4]] = uk[1:-3].reshape(-1, 1)   # b1*u[k-1]
        A[:, [5]] = uk[2:-2].reshape(-1, 1)   # b2*u[k-2]
        A[:, [6]] = uk[3:-1].reshape(-1, 1)   # b3*u[k-3]
        A[:, [7]] = uk[4:].reshape(-1, 1)     # b4*u[k-4]   
        
    else:
        raise ValueError(f"Order {order} not implemented yet")
    
    #Check matrix conditioning
    # condition_number = np.linalg.cond(A)
    # rank = np.linalg.matrix_rank(A)
    #print("Condition number of A: ", condition_number)
    #print("Rank of A: ", rank)
    
    return A, b, u_bar, y_bar
#%%

# Error analysis
def compute_cross_dataset_error(cross_dataset_data):
    results = {}
    print("=== Cross-Dataset Model Validation Results ===")
    print(f"{'Order':<6} {'Train MSE':<12} {'Train MSO':<12} {'Train NMSE':<12} {'Test MSE (avg)':<15} {'Test MSO (avg)':<15} {'Test NMSE (avg)':<15}")
    print("-" * 100)
    
    # Extract training data
    uk_train = cross_dataset_data['train']['input']
    yk_train = cross_dataset_data['train']['output']
    u_bar = cross_dataset_data['scaling']['u_bar']
    y_bar = cross_dataset_data['scaling']['y_bar']
    
    for order in [1, 2, 3]:
        # Get A_train and b_train for this order using unified function
        A_train, b_train, _, _ = A_matrix(uk_train, yk_train, u_bar, y_bar, order)
        
        # Solve for x using training data
        x = linalg.solve(A_train.T @ A_train, A_train.T @ b_train)
        
        # Training error
        residuals_train = b_train - A_train @ x
        mse_train = np.mean(residuals_train**2)
        mso_train = np.mean(b_train**2)
        train_nmse = mse_train / mso_train
        
        # Test on each test dataset
        test_mse_list = []
        test_mso_list = []
        test_nmse_list = []
        
        for test_idx, test_data in cross_dataset_data['test'].items():

            uk_test = test_data['input']
            yk_test = test_data['output']
            
            # Get A_test and b_test for this order using unified function
            A_test, b_test, _, _ = A_matrix(uk_test, yk_test, u_bar, y_bar, order)
            
            residuals_test = b_test - A_test @ x
            mse_test = np.mean(residuals_test**2)
            mso_test = np.mean(b_test**2)
            nmse_test = mse_test / mso_test
            
            test_mse_list.append(mse_test)
            test_mso_list.append(mso_test)
            test_nmse_list.append(nmse_test)
        
        # Average across test datasets
        avg_test_mse = np.mean(test_mse_list)
        avg_test_mso = np.mean(test_mso_list)
        avg_test_nmse = np.mean(test_nmse_list)
        
        results[order] = {
            'train_nmse': train_nmse,
            'test_nmse': avg_test_nmse,
            'mse_train': mse_train,
            'mso_train': mso_train,
            'mse_test': avg_test_mse,
            'mso_test': avg_test_mso,
            'x': x,
            'A_train': A_train,
            'b_train': b_train
        }
        
        print(f"{order:<6} {mse_train:<12.8f} {mso_train:<12.8f} {train_nmse:<12.8f} {avg_test_mse:<15.8f} {avg_test_mso:<15.8f} {avg_test_nmse:<15.8f}")
    
    return results

# Run cross-dataset validation

# Store results for averaging
all_results = []

for i in range(4):
    cross_dataset_data = cross_dataset_split(train_dataset=i)
    print(f"Results for train dataset {i+1}:")
    print("-" * 50)
    results = compute_cross_dataset_error(cross_dataset_data)
    all_results.append(results)
    print("-" * 50)

# Simple average summary
print("\nAVERAGE ACROSS ALL TRAINING DATASETS:")
print(f"{'Order':<6} {'Avg Train NMSE':<15} {'Avg Test NMSE':<15} {'Avg Train MSE':<15} {'Avg Test MSE':<15} {'Avg Train MSO':<15} {'Avg Test MSO':<15}")
print("-" * 40)

for order in [1, 2, 3]:
    train_nmse_avg = np.mean([all_results[i][order]['train_nmse'] for i in range(4)])
    test_nmse_avg = np.mean([all_results[i][order]['test_nmse'] for i in range(4)])
    train_mse_avg = np.mean([all_results[i][order]['mse_train'] for i in range(4)])
    test_mse_avg = np.mean([all_results[i][order]['mse_test'] for i in range(4)])
    train_mso_avg = np.mean([all_results[i][order]['mso_train'] for i in range(4)])
    test_mso_avg = np.mean([all_results[i][order]['mso_test'] for i in range(4)])
    print(f"{order:<6} {train_nmse_avg:<15.6f} {test_nmse_avg:<15.6f} {train_mse_avg:<15.6f} {test_mse_avg:<15.6f} {train_mso_avg:<15.6f} {test_mso_avg:<15.6f}")
    print("-" * 100)


