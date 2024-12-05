import numpy as np
import os
import matplotlib.pyplot as plt
import LNP_model as lnp
import MAP as map
import MCMC as mcmc
import stimulus_filter_poly as sfp
import pickle

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'Data/')
figure_path = os.path.join(current_path, 'Figures/')

def plot_lnp_model(lnp_filename):
    run_specifics, t, x, r, lambda_, u = lnp.npz_load(lnp_filename)
    dt, t, num_neurons, cell_type, dc_firing = lnp.unravel_run_specifics(run_specifics)

    # Plot LNP Example Usage
    plt.figure(figsize=(10,5))  

    # Plot a dot when there is a spike
    for neuron_idx in range(num_neurons):
        spike_times = t[r[neuron_idx] == 1]
        if cell_type[neuron_idx] == 'on':
            plt.scatter(spike_times, np.full_like(spike_times, 2 + neuron_idx * 0.25), color='red')
            plt.plot(t, np.sqrt(3)*np.array(u[neuron_idx]/np.max(u[neuron_idx])), color='red', linestyle='--')
        elif cell_type[neuron_idx] == 'off':
            plt.scatter(spike_times, np.full_like(spike_times, 2 + neuron_idx * 0.25), color='blue')
            plt.plot(t, np.sqrt(3)*np.array(u[neuron_idx]/np.max(u[neuron_idx])), color='blue', linestyle='--')


    plt.scatter([], [], color='red', label='ON Spike')
    plt.scatter([], [], color='blue', label='OFF Spike')
    plt.plot([], [], color='red', label='ON Log Firing Rate (norm.)')
    plt.plot([], [], color='blue', label='OFF Log Firing Rate (norm.)')

    plt.step(t, x, color='black', label='Stimulus', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Stimulus [a.u.]')
    # Plotting two horizontal lines to represent the range of the stimulus
    plt.axhline(y=np.sqrt(3), color='black', linestyle='--')
    plt.axhline(y=-np.sqrt(3), color='black', linestyle='--')

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the right margin
    plt.show()
    #plt.savefig(folder_path + 'lnp_example.png')

def plot_x_comparison(lnp_filename, x_new, x_new_name):
    run_specifics, t, x_og, r, lambda_, u = lnp.npz_load(lnp_filename)
    dt, t, num_neurons, cell_type, dc_firing = lnp.unravel_run_specifics(run_specifics)

    # Plot LNP Example Usage
    plt.figure(figsize=(10,5))  

    # Plot a dot when there is a spike
    for neuron_idx in range(num_neurons):
        spike_times = t[r[neuron_idx] == 1]
        if cell_type[neuron_idx] == 'on':
            plt.scatter(spike_times, np.full_like(spike_times, 2 + neuron_idx * 0.25), color='red')
        elif cell_type[neuron_idx] == 'off':
            plt.scatter(spike_times, np.full_like(spike_times, 2 + neuron_idx * 0.25), color='blue')


    plt.scatter([], [], color='red', label='ON Spike')
    plt.scatter([], [], color='blue', label='OFF Spike')

    plt.step(t, x_og, color='black', label='X_original', linewidth=2, linestyle='--')
    plt.step(t, x_new, color='orange', label=x_new_name, linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Stimulus [a.u.]')
    # Plotting two horizontal lines to represent the range of the stimulus
    plt.axhline(y=np.sqrt(3), color='black', linestyle='--')
    plt.axhline(y=-np.sqrt(3), color='black', linestyle='--')
    msq_error = compute_msq_error(x_og, x_new)
    plt.title(f'{x_new_name} (MSQ Error: {msq_error:.4f})')

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the right margin
    plt.show()
    #plt.savefig(folder_path + 'lnp_example.png')

def plot_comparisons():
    lnp_filename = 'lnp_4cell_DC0.pkl'

    x_map = [-0.73126245, 1.73205081, 1.11479213, -1.73205081, 1.64890657, -1.3572254,
    -1.73205081, 0.0424869, 1.73205081, -0.64697451, -1.2732158, 0.44423704,
    1.47060897, -0.6595857, -1.18982932, -1.73205081, -0.2725394, 0.10897563,
    -0.98644954, 1.73205081, -0.55845219, -1.73205081, 1.26835019, -1.32178553,
    0.59755829, -1.05089793, 0.68102227, -1.28095576, 0.93445336, -1.5123613,
    1.22118337, -1.73205081, 0.61888552, 1.10299167, -0.09959838, 1.73205081,
    -0.24636779, -0.7370951, 1.37303803, -0.87307996, 1.20372655, -0.98271209,
    1.48790924, -1.23691154, 0.60303151, 1.73205081, -0.56205255, -0.15816603,
    -1.73205081, 1.03019795]

    # x_bls = [-1.48983269, 0.43667066, 0.33931943, 0.23762527, 0.64899324, 0.96498377,
    #  -0.09269755, -0.643588, 0.8013906, -0.69176414, 0.15626003, -1.41495117,
    #  -1.34803341, 1.14830494, -0.1139146, 0.65335995, -0.82939713, -0.60116855,
    #  -1.33418336, -0.90138877, -0.08699605, 0.93142084, 0.07467726, -0.79067425,
    #  1.05903755, 0.66128977, 0.82885614, -0.21391548, 0.03404697, -1.35051888,
    #  -1.23314957, 1.15700821, -1.51563448, -0.67278945, -0.65669662, -1.56431713,
    #  1.21521295, 0.05318379, -0.79608456, -1.24478407, 0.24637589, -0.43969146,
    #  -1.50256721, 0.35008919, -0.81489129, 1.20441335, 0.57292906, -1.40719946,
    #  -0.00757154, 0.43488701] # M=100, B=10

    x_bls = [-0.53558634, 1.15614736, 0.53422033, 0.39575979, -0.07303699, -0.91046803,
    -0.02595552, -0.94547217, 1.42673744, -0.82045363, -0.18409594, -0.10549827,
        0.03316057, 0.96503119, -1.29428959, -0.72866621, -1.11739978, 0.17661719,
    -0.94991943, -1.12138329, 0.57878146, 0.00346207, -0.90335128, -0.66206174,
        0.01404471, 0.84163995, 0.16168196, -1.06961527, 0.10955881, -1.16784496,
    -1.04299031, -0.42647704, -1.180789, 1.01771318, 0.06869074, -0.64018847,
        0.9038675, -1.18746004, -0.04251257, 0.00922343, 0.77331149, -1.23565137,
    -0.9473217, 0.13645771, -0.36481019, 1.06358262, 0.17132541, -1.26851607,
        0.12316311, 0.30090259] # M=500, B=50

    #plot_x_comparison(lnp_filename, x_map, 'X_map')
    #plot_x_comparison(lnp_filename, x_bls, 'X_bls')

def plot_computation():
    msq_map = 1.78
    map_computation_time = 18.45258 # [hr]

    bls = [10, 50, 100, 500]
    msq_bls = [0.5, 0.6, 0.7, 0.8]
    bls_computation_time = np.array([60, 600, 800, 1000])/3600 # [hr]

    # Create a figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Bar graph for MSQ errors
    methods = ['MAP'] + [f'BLS (N={n})' for n in bls]
    msq_errors = [msq_map] + msq_bls
    ax1.axhline(y=2.0711495053913445, color='red', linestyle='--', label='Random MSE')
    ax1.legend()
    ax1.bar(methods, msq_errors, color=['blue'] + ['orange']*len(bls))
    ax1.set_title('Mean Squared Errors (MSE)')
    ax1.set_ylabel('MSE Error')
    ax1.set_xlabel('Method')

    # Bar graph for computation times
    computation_times = [map_computation_time] + list(bls_computation_time)
    ax2.bar(methods, computation_times, color=['blue'] + ['orange']*len(bls))
    ax2.set_title('Computation Times')
    ax2.set_ylabel('Time (hours)')
    ax2.set_xlabel('Method')

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

plot_computation()