"""MECH 412 sample code.

J R Forbes, 2025/10/13

This code loads the data.
"""

# %%
# Libraries

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import pathlib
from scipy import linalg
import control
from scipy.linalg import expm, logm

#Credit to J R Forbes for the d2c function#
def d2c(Pd):
    '''
    Parameters
    ----------
    Pd : Discrete-time (DT) transfer function (TF) from control
        package computed uzing a zoh.
        A DT TF to be converted to a continuous-time (CT) TF.

    Returns
    -------
    Pc : A CT TF computed from the DT TF given.

    References
    -------
    K. J. Astrom and B. Wittenmark, Computer Controlled Systems:
        Theory and Design, 3rd., Prentice-Hall, Inc., 1997, pp. 32-37.
    '''
    # Preliminary calculations
    dt = Pd.dt  # time step
    Pd_ss = control.ss(Pd)  # Convert Pd(z) TF to state-space (SS) realization
    Ad, Bd, Cd, Dd = Pd_ss.A, Pd_ss.B, Pd_ss.C, Pd_ss.D  # Extract SS matrices
    n_x, n_u = Ad.shape[0], Bd.shape[1]  # Extract shape of SS matrices

    # Form the matrix Phi, which is composed of Ad and Bd
    Phi1 = np.hstack([Ad, Bd])
    Phi2 = np.hstack([np.zeros([n_u, n_x]), np.eye(n_u)])
    Phi = np.vstack([Phi1, Phi2])

    # Compute Upsilon the matrix log of Phi
    Upsilon = logm(Phi) / dt

    # Extract continuous-time Ac and Bc
    # (Recall, a SS realization is *not* unique. The matrices extracted
    # from Upsilon may not equal Ac and Bc in some canonical form.)
    Ac = Upsilon[:n_x, :n_x]
    Bc = Upsilon[:n_x, (n_x - n_u + 1):]

    # The continuous-time Cc and Cc equal the discrete-time Cd and Dd
    Cc, Dc = Cd, Dd

    # Compute the transfer function Pc(s)
    Pc = control.ss2tf(Ac, Bc, Cc, Dc)
    return Pc


# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# path = pathlib.Path('figs')
# path.mkdir(exist_ok=True)


# %%
# Read in all input-output (IO) data
path = pathlib.Path('/Users/dylanmyers/Desktop/5thyear/MECH412/Project/load_data_sc/PRBS_DATA')
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

# %%
# Load a dataset
k_train = 2
data_read = data[k_train, :, :]

# Extract time
t = data_read[:, 0]
N = t.size
T = t[1] - t[0]

# Extract input and output
u_raw = data_read[:, 1]  # V, volts
y_raw = data_read[:, 2]  # LPM, liters per minute


def plot_data(k_train, t, u_raw, y_raw):
    fig, ax = plt.subplots(2, 1)
    ax[0].set_ylabel(r'$u(t)$ (V)')
    ax[1].set_ylabel(r'$y(t)$ (LPM)')
    # Plot data
    ax[0].plot(t, u_raw, label='input', color='C0')
    ax[1].plot(t, y_raw, label='output', color='C1')
    for a in np.ravel(ax):
        a.set_xlabel(r'$t$ (s)')
        a.legend(loc='best')
    fig.tight_layout()
    # This ell variable will allow you to save a plot with the number ell in the plot name
    ell = k_train
    return None
    # fig.savefig('test_plot_%s.pdf' % ell)
plot_data(k_train, t, u_raw, y_raw)
#plt.show()
# %%
# %%
#Conditioning and Building, for a second order system

#Nominal data conditioning, treating the entire data set as one single input-output data set with no split. Normalizes the data to be between 0 and 1.
def no_ksplit():
    u_bar = np.max([np.max(u_raw), np.abs(np.min(u_raw))]) # maximum input
    y_bar = np.max([np.max(y_raw), np.abs(np.min(y_raw))]) # maximum output
    # Normalize data and flip "up down"
    uk = u_raw[::-1] / u_bar
    yk = y_raw[::-1] / y_bar



    # Since no split, test data is same as training data
    uk_test = uk
    yk_test = yk

    return uk, yk, u_bar, y_bar, uk_test, yk_test

#Splitting data into k-fold cross validation, learned from ecse 551**** MIGHT BE OUT OF SCOPE
def k_split(k):
    uk = u_raw[k:N] / u_bar
    yk = y_raw[k:N] / y_bar
    #normalize the data 
        # Normalize data and flip "up down"
    uk = u_raw[::-1] / u_bar
    yk = y_raw[::-1] / y_bar
    return uk, yk, u_bar, y_bar



#This splits a given dataset into training and test, also normalizes the data.
def train_test_split():
    # Split data into 2 equal halves
    mid = len(u_raw) // 2
    # Split the data
    u_train = u_raw[:mid]
    y_train = y_raw[:mid]
    u_test = u_raw[mid:]
    y_test = y_raw[mid:]
    
    # Use only training data for normalization
    u_bar = np.max([np.max(u_train), np.abs(np.min(u_train))])
    y_bar = np.max([np.max(y_train), np.abs(np.min(y_train))])

    # Normalize both train and test and flip
    uk_train = u_train[::-1] / u_bar
    yk_train = y_train[::-1] / y_bar
    uk_test = u_test[::-1] / u_bar
    yk_test = y_test[::-1] / y_bar
    
    return uk_train, yk_train, u_bar, y_bar, uk_test, yk_test


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



# Choose your data split method:
#uk, yk, u_bar, y_bar, uk_test, yk_test = no_ksplit()  # Change this to test different splits
#uk, yk, u_bar, y_bar, uk_test, yk_test = train_test_split()  # Split single dataset by time

# Cross-dataset split: train on dataset 0, test on datasets 1, 2, 3
cross_dataset_data = cross_dataset_split(train_dataset=2)
uk = cross_dataset_data['train']['input']
yk = cross_dataset_data['train']['output']
u_bar = cross_dataset_data['scaling']['u_bar']
y_bar = cross_dataset_data['scaling']['y_bar']

#Inspired by lecture notes 7a-sysID# --> CONSIDER REFACTORING <---

# 1st Order System: y[k] = -a1*y[k-1] + b1*u[k-1]
def A_matrix_1stOrder(uk, yk, u_bar, y_bar):

    # Form A and B matrices for system ID
    N_train = len(yk)  # Use length of training data
    b = yk[:-1].reshape(-1, 1)
    A = np.zeros((N_train - 1, 2))
    A[:, [0]] = -yk[1:].reshape(-1, 1)  # -a1*y[k-1]
    A[:, [1]] = uk[1:].reshape(-1, 1)   # b1*u[k-1]

    #Checking the A matrix
    condition_number = np.linalg.cond(A)
    #print("1st Order - Condition number (A): ", condition_number)
    rank = np.linalg.matrix_rank(A)
    #print("1st Order - Rank of A: ", rank)

    return A, b, u_bar, y_bar

# 2nd Order System: y[k] = -a1*y[k-1] - a2*y[k-2] + b1*u[k-1] + b2*u[k-2]
def A_matrix_2ndOrder(uk, yk, u_bar, y_bar):

    # Form A and B matrices for system ID
    N_train = len(yk)  # Use actual length of training data
    b = yk[:-2].reshape(-1, 1)
    A = np.zeros((N_train - 2, 4))
    A[:, [0]] = -yk[1:-1].reshape(-1, 1)  # -a1*y[k-1]
    A[:, [1]] = -yk[2:].reshape(-1, 1)    # -a2*y[k-2]
    A[:, [2]] = uk[1:-1].reshape(-1, 1)   # b1*u[k-1]
    A[:, [3]] = uk[2:].reshape(-1, 1)     # b2*u[k-2]

    #Checking the A matrix
    condition_number = np.linalg.cond(A)
    #print("2nd Order - Condition number (A): ", condition_number)
    rank = np.linalg.matrix_rank(A)
    #print("2nd Order - Rank of A: ", rank)

    return A, b, u_bar, y_bar


# 3rd Order System: y[k] = -a1*y[k-1] - a2*y[k-2] - a3*y[k-3] + b1*u[k-1] + b2*u[k-2] + b3*u[k-3]
def A_matrix_3rdOrder(uk, yk, u_bar, y_bar):
    # Form A and B matrices for system ID
    N_train = len(yk)  # Use actual length of training data
    b = yk[:-3].reshape(-1, 1)
    A = np.zeros((N_train - 3, 6))
    A[:, [0]] = -yk[1:-2].reshape(-1, 1)  # -a1*y[k-1]
    A[:, [1]] = -yk[2:-1].reshape(-1, 1)  # -a2*y[k-2]
    A[:, [2]] = -yk[3:].reshape(-1, 1)    # -a3*y[k-3]
    A[:, [3]] = uk[1:-2].reshape(-1, 1)   # b1*u[k-1]
    A[:, [4]] = uk[2:-1].reshape(-1, 1)   # b2*u[k-2]
    A[:, [5]] = uk[3:].reshape(-1, 1)     # b3*u[k-3]

    #Checking the A matrix
    condition_number = np.linalg.cond(A)
    #print("3rd Order - Condition number (A): ", condition_number)
    rank = np.linalg.matrix_rank(A)
    #print("3rd Order - Rank of A: ", rank)

    return A, b, u_bar, y_bar

def A_matrix_4thOrder(uk, yk, u_bar, y_bar):
    # Form A and B matrices for system ID
    N_train = len(yk)  # Use actual length of training data
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

    #Checking the A matrix
    condition_number = np.linalg.cond(A)
    #print("4th Order - Condition number (A): ", condition_number)
    rank = np.linalg.matrix_rank(A)
    #print("4th Order - Rank of A: ", rank)

    return A, b, u_bar, y_bar


# %%
# Solving for TF coefficients
def solve_Ax_b(order, uk, yk, u_bar, y_bar):

    # Get A matrix and data based on order
    if order == 1:
        A, b, u_bar, y_bar = A_matrix_1stOrder(uk, yk, u_bar, y_bar)
    elif order == 2:
        A, b, u_bar, y_bar = A_matrix_2ndOrder(uk, yk, u_bar, y_bar)
    elif order == 3:
        A, b, u_bar, y_bar = A_matrix_3rdOrder(uk, yk, u_bar, y_bar)
    else:
        raise ValueError(f"Order {order} not implemented yet")
    
    # Solve Ax = b problem
    x = linalg.solve(A.T @ A, A.T @ b)

    # Compute TF coefficients
    Pd_ID_den = np.hstack([1, x[0:order, :].reshape(-1,)]) # den coefficients of DT TF
    Pd_ID_num = x[order:, :].reshape(-1,) # num coefficients of DT TF
    
    # Create discrete-time transfer function DONT FORGET TO DENOMMALIZE
    Pd_ID = y_bar / u_bar * control.tf(Pd_ID_num, Pd_ID_den, T)

    # Convert to continuous-time
    Pc_ID = d2c(Pd_ID)
    return Pd_ID, Pc_ID



# %%
#Model Order Copmutation +  Validation
def get_models(uk, yk, u_bar, y_bar):
    models = {}
    for order in [1, 2, 3]:
        P_d, P_c = solve_Ax_b(order, uk, yk, u_bar, y_bar)
        models[order] = {'discrete': P_d, 'continuous': P_c}
    return models

def compute_models():
    models = get_models(uk, yk, u_bar, y_bar)
    
    
    for order in sorted(models):
        print(f"\nOrder {order} System:")
        print("Discrete-time TF (Pd):")
        print(models[order]['discrete'])
        print("Continuous-time TF (Pc):")
        print(models[order]['continuous'])

    

    return models

compute_models()

#%%

def plot_model_comparison(models, uk, yk, u_bar, y_bar, uk_test, yk_test, t):
    """
    Create clean plots showing input, actual output, and identified output
    for each model order, similar to the reference image.
    """
    import matplotlib.pyplot as plt
    
    # Unnormalize the data for plotting
    u_actual = uk[::-1] * u_bar  # Flip back and unnormalize
    y_actual = yk[::-1] * y_bar  # Flip back and unnormalize
    u_test_actual = uk_test[::-1] * u_bar
    y_test_actual = yk_test[::-1] * y_bar
    
    # Create time vectors - use proper indexing
    t_train = t[:len(u_actual)]
    t_test = t[len(u_actual):len(u_actual)+len(u_test_actual)]
    
    # Ensure time vectors are not empty
    if len(t_train) == 0:
        t_train = np.linspace(0, len(u_actual)-1, len(u_actual))
    if len(t_test) == 0:
        t_test = np.linspace(0, len(u_test_actual)-1, len(u_test_actual))
    
    # Create subplots for each model order
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('System Identification Results', fontsize=16)
    
    for i, order in enumerate([1, 2, 3]):
        # Get the discrete transfer function
        P_d = models[order]['discrete']
        
        # Debug: Print transfer function for each order
        print(f"\nOrder {order} Transfer Function:")
        print(f"Numerator: {P_d.num}")
        print(f"Denominator: {P_d.den}")
        
        # Simulate response to training input
        t_train_sim, y_train_sim = control.forced_response(P_d, t_train, u_actual)
        
        # Plot all three signals on the same subplot
        axes[i].plot(t_train, u_actual, 'b--', label='input', linewidth=2)
        axes[i].plot(t_train, y_actual, 'orange', label='output', linewidth=2)
        axes[i].plot(t_train_sim, y_train_sim, 'g:', label='IDed output', linewidth=2)
        
        # Formatting
        axes[i].set_title(f'Model Order {order}', fontsize=14)
        axes[i].set_xlabel('t (s)', fontsize=12)
        axes[i].set_ylabel('u(t) and y(t) (units)', fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Set consistent y-axis limits for better comparison
        y_min = min(np.min(u_actual), np.min(y_actual), np.min(y_train_sim))
        y_max = max(np.max(u_actual), np.max(y_actual), np.max(y_train_sim))
        axes[i].set_ylim(y_min - 0.5, y_max + 0.5)
    
    plt.tight_layout()
    plt.show()
    
    return fig


#%%

#Error analysis

#function to obtain our A_test
def get_Atest(order, uk_test, yk_test, u_bar, y_bar):
    models = {}
    for order in [1, 2, 3]:
        if order == 1:
            A, b, u_bar, y_bar = A_matrix_1stOrder(uk_test, yk_test, u_bar, y_bar)
        elif order == 2:
            A, b, u_bar, y_bar = A_matrix_2ndOrder(uk_test, yk_test, u_bar, y_bar)
        elif order == 3:
            A, b, u_bar, y_bar = A_matrix_3rdOrder(uk_test, yk_test, u_bar, y_bar)
        models[order] = {'A': A, 'b': b, 'u_bar': u_bar, 'y_bar': y_bar}
    return models


#For single datase (split) validation
def compute_error(uk, yk, u_bar, y_bar, uk_test, yk_test):
    results = {}
    print("=== Model Validation Results (Normalized) ===")
    print(f"{'Order':<6} {'Train MSE':<12} {'Train MSO':<12} {'Train NMSE':<12} {'Test MSE':<12} {'Test MSO':<12} {'Test NMSE':<12}")
    print("-" * 90)
    
    for order in [1, 2, 3]:
        # Get A_train and b_train for this order
        if order == 1:
            A_train, b_train, u_bar, y_bar = A_matrix_1stOrder(uk, yk, u_bar, y_bar)
        elif order == 2:
            A_train, b_train, u_bar, y_bar = A_matrix_2ndOrder(uk, yk, u_bar, y_bar)
        elif order == 3:
            A_train, b_train, u_bar, y_bar = A_matrix_3rdOrder(uk, yk, u_bar, y_bar)
        
        # Solve for x using training data: A_train * x = b_train , should stay the same for both test and train computations.
        x = linalg.solve(A_train.T @ A_train, A_train.T @ b_train)

        # Get A_test and b_test for this order, this will build the A test matrix for each of the 3 orders
        test_models = get_Atest(order, uk_test, yk_test, u_bar, y_bar)
        A_test = test_models[order]['A']
        #b_test is simply the test split of the output data.
        b_test = yk_test[:-order].reshape(-1, 1)
        
        residuals_train = b_train - A_train @ x
        residuals_test = b_test - A_test @ x

        # Calculate normalized errors using u_bar and y_bar

        # MSE = (1/N) * ||residuals||²
        mse_train = np.mean(residuals_train**2)
        mse_test = np.mean(residuals_test**2)
        
        # MSO = (1/N) * ||b||² 
        mso_train = np.mean(b_train**2)
        mso_test = np.mean(b_test**2) 
        
        # NMSE = MSE / MSO
        train_nmse = mse_train / mso_train
        test_nmse = mse_test / mso_test

        # Store these values
        results[order] = {
            'train_nmse': train_nmse,
            'test_nmse': test_nmse,
            'mse_train': mse_train,
            'mso_train': mso_train,
            'mse_test': mse_test,
            'mso_test': mso_test,
            'x': x,
            'A_train': A_train,
            'b_train': b_train,
            'A_test': A_test,
            'b_test': b_test
        }
        
        print(f"{order:<6} {mse_train:<12.8f} {mso_train:<12.8f} {train_nmse:<12.8f} {mse_test:<12.8f} {mso_test:<12.8f} {test_nmse:<12.8f}")
    
    return results

# For cross-dataset validation
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
        # Get A_train and b_train for this order, could potentially make this more efficient in the future (O(1))
        if order == 1:
            A_train, b_train, _, _ = A_matrix_1stOrder(uk_train, yk_train, u_bar, y_bar)
        elif order == 2:
            A_train, b_train, _, _ = A_matrix_2ndOrder(uk_train, yk_train, u_bar, y_bar)
        elif order == 3:
            A_train, b_train, _, _ = A_matrix_3rdOrder(uk_train, yk_train, u_bar, y_bar)
        
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
            
            # Get A_test and b_test for this order
            if order == 1:
                A_test, b_test, _, _ = A_matrix_1stOrder(uk_test, yk_test, u_bar, y_bar)
            elif order == 2:
                A_test, b_test, _, _ = A_matrix_2ndOrder(uk_test, yk_test, u_bar, y_bar)
            elif order == 3:
                A_test, b_test, _, _ = A_matrix_3rdOrder(uk_test, yk_test, u_bar, y_bar)
            
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
print(f"{'Order':<6} {'Avg Train NMSE':<15} {'Avg Test NMSE':<15}")
print("-" * 40)

for order in [1, 2, 3]:
    train_nmse_avg = np.mean([all_results[i][order]['train_nmse'] for i in range(4)])
    test_nmse_avg = np.mean([all_results[i][order]['test_nmse'] for i in range(4)])
    print(f"{order:<6} {train_nmse_avg:<15.6f} {test_nmse_avg:<15.6f}")


# %%
# Plot model comparisons
#models = compute_models()
#plot_model_comparison(models, uk, yk, u_bar, y_bar, uk_test, yk_test, t)



