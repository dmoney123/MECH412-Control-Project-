import sys
import os
import control
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Param_Opt import construct_TF, optimal_parameters
from src.uncertainty_bound import residuals, residual_max_mag, upperbound


order = 2
w_shared = np.logspace(-2, 3, 1000)

def model_construct(order):
    # Initial model construction + sanity check
    analysis = optimal_parameters(order)
    functions = construct_TF(order, analysis)
    #Creating lists of off nominal transfer functions
    p_off_nominal = []
    for name, tfdata in functions.items():
        if name != 'optimal':
            p_off_nominal.append(tfdata['continuous'])
    #Assigning the nominal transfer function
    p_nominal = functions['optimal']['continuous']
    return p_off_nominal, p_nominal

def plot_1():
    p_off_nominal, p_nominal = model_construct(order)
    W2_list = W2()
    R = residuals(p_nominal, p_off_nominal)
    #sanity check, this should be zero
    R_nom = residuals(p_nominal, [p_nominal])  # residual of nominal with itself

    #start with 10E-2 to 10E3 rad/s with 500 points, change as progression
    mag_max_dB, mag_max_abs = residual_max_mag(R, w_shared)

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].set_title('Residual Magnitude Plot - MECH412 Uncertainty Analysis', fontsize=14, fontweight='bold')
    ax[1].set_title('Residual Magnitude Plot - MECH412 Uncertainty Analysis', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('Frequency (rad/s)', fontsize=12)
    ax[1].set_xlabel('Frequency (rad/s)', fontsize=12)
    ax[0].set_ylabel('Magnitude (dB)', fontsize=12)
    ax[1].set_ylabel('Magnitude (absolute)', fontsize=12)
    ax[0].grid(True, alpha=0.3, linestyle='--')
    ax[1].grid(True, alpha=0.3, linestyle='--')


    # Define colors and labels for each residual
    colors = ['blue', 'green', 'red', 'orange']
    labels = ['R₁ (Dataset 0)', 'R₂ (Dataset 1)', 'R₃ (Dataset 2)', 'R₄ (Dataset 3)']


    #Pks
    for i in range(len(p_off_nominal)):
        mag_abs, _, _ = control.frequency_response(R[i], w_shared)
        mag_dB = 20 * np.log10(mag_abs)
        
        # Use different colors and labels for each residual
        color = colors[i] if i < len(colors) else f'C{i}'
        label = labels[i] if i < len(labels) else f'R{i+1}'
        
        # Magnitude plot (dB)
        ax[0].semilogx(w_shared, mag_dB, '--', color=color, linewidth=2, alpha=0.8, label=label)
        # Magnitude plot (absolute)
        ax[1].semilogx(w_shared, mag_abs, '--', color=color, linewidth=2, alpha=0.8, label=label)
        # Plot the upper bound


    #W2_list
    w2_colors = ['crimson', 'slateblue']  # Only 2 colors for degrees 3, 4
    for i in range(len(W2_list)):
        mag_abs, _, _ = control.frequency_response(W2_list[i], w_shared)
        mag_dB = 20 * np.log10(mag_abs)
        color = w2_colors[i] if i < len(w2_colors) else f'C{i+4}'
        degree = i + 3  # degrees 3, 4
        ax[0].semilogx(w_shared, mag_dB, '-', color=color, linewidth=2, label=f'W2(s) degree {degree}')
        ax[1].semilogx(w_shared, mag_abs, '-', color=color, linewidth=2, label=f'W2(s) degree {degree}')
    
    #upperbound
    ax[0].semilogx(w_shared, mag_max_dB, '-', color='C4', label='upper bound', linewidth=2)
    ax[1].semilogx(w_shared, mag_max_abs, '-', color='C4', label='upper bound', linewidth=2)

    #pnom sanity check - should be zero everywhere
    mag_optimal, _, _ = control.frequency_response(R_nom[0], w_shared) 
    mag_optimal_dB = 20 * np.log10(mag_optimal)
    ax[0].semilogx(w_shared, mag_optimal_dB, '-', color='black', linewidth=2, label='Nominal (sanity check)')
    ax[1].semilogx(w_shared, mag_optimal, '-', color='black', linewidth=2, label='Nominal (sanity check)')

    ax[0].legend(loc='best', fontsize=10)
    ax[1].legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
    plt.show()



def W2():
    p_off_nominal, p_nominal = model_construct(order)
    R = residuals(p_nominal, p_off_nominal)
    mag_max_dB, mag_max_abs = residual_max_mag(R, w_shared)

    W2_list = []
    for i in range(3, 5):  # degrees 3, 4 only (lower degrees don't work well)
        W2_list.append(upperbound(w_shared, mag_max_abs, i))
    return W2_list


plot_1()             








