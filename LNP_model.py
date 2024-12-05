import numpy as np
import os
import matplotlib.pyplot as plt
import stimulus_filter_poly as sfp
import pickle

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'Data/')
figure_path = os.path.join(current_path, 'Figures/')

k_constant = 15

def compute_u_i(t, x, cell_type_i, dc_firing):

    if cell_type_i == 'off':
        term1 = 3.1
    elif cell_type_i == 'on':
        term1 = 2.25

    if not(dc_firing):
        term1 = 0
        
    term3 = 0
    
    dt = t[1] - t[0]
    tau = 0.25 # [s]
    t_sum = np.arange(0,tau,dt)

    # plt.figure()
    # plt.plot(0-t_sum, sfp.extract_filter(cell_type_i, 'k', 0-t_sum))
    # plt.show()
    u_i = []
    for i, t_i in enumerate(t):
        term2 = 0
        for j, t_j in enumerate(t_sum):
            #print(f'k time', 0 - t_j)
            #print(f'x time', t_i - t_j)
            term2 += k_constant * sfp.extract_filter_t(cell_type_i, 'k', 0 - t_j) * extract_x_t(t, x, t_i - t_j)
        u_i.append(term1 + term2 + term3)

    return u_i

def compute_lambda_i(u_i):
    return np.exp(u_i)

def compute_lambda_(run_specifics, x):
    dt, t, num_neurons, cell_type, dc_firing = unravel_run_specifics(run_specifics)

    lambda_ = []
    for i in range(num_neurons):
        u_i = compute_u_i(t, x, cell_type[i], dc_firing)
        lambda_i = compute_lambda_i(u_i)
        lambda_.append(lambda_i)

    return np.array(lambda_)

def compute_r_i(t, lambda_i):
    # probabilistic spiking extraction from lambda_i
    dt = t[1] - t[0]

    r_i = np.zeros_like(lambda_i)
    for t in range(len(lambda_i)):
        r_i[t] = (np.random.rand()) < (lambda_i[t]*dt)
    return r_i

def extract_x_t(t, x, t_i):
    if t_i < 0:
        return 0
    else:
        index = np.argmin(np.abs(t - t_i))
        return x[index]

def unravel_run_specifics(run_specifics):
    dt = run_specifics['dt']
    t = run_specifics['t']
    num_neurons = run_specifics['num_neurons']
    cell_type = run_specifics['cell_type']
    dc_firing = run_specifics['dc_firing']

    return dt, t, num_neurons, cell_type, dc_firing

def compute_r(run_specifics, x, plot=False):
    dt, t, num_neurons, cell_type, dc_firing = unravel_run_specifics(run_specifics)
   
    r = []
    lambda_ = []
    u = []
    for i in range(num_neurons):
        u_i = compute_u_i(t, x, cell_type[i], dc_firing)
        lambda_i = compute_lambda_i(u_i)
        r_i = compute_r_i(t, lambda_i)
        lambda_.append(lambda_i)
        u.append(u_i)
        r.append(r_i)


    if plot:
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

    return np.array(r), np.array(lambda_), np.array(u)

def npz_save(run_specifics, t, x, r, lambda_, u, filename):
    print('Saving Data')
    data = {
        'run_specifics': run_specifics,
        't': t,
        'x': x,
        'r': r,
        'lambda_': lambda_,
        'u': u
    }

    file_path = os.path.join(data_path, filename)
    np.savez(file_path, run_specifics=run_specifics, t=t, x=x, r=r, lambda_=lambda_, u=u)
    print('Saved!')

def npz_load(filename):
    print('Loading Data')
    file_path = os.path.join(data_path, filename)
    data = np.load(file_path, allow_pickle=True)
    run_specifics = data['run_specifics']
    t = data['t']
    x = data['x']
    r = data['r']
    lambda_ = data['lambda_']
    u = data['u']
    print('Loaded!')
    return run_specifics, t, x, r, lambda_, u

def compute_msq_error(x_og, x_new):
    return np.mean((x_og - x_new)**2)

def run_lnp(dt, T, num_neurons, dc_firing, save=False, plot=False):
    # Example Usage
    t = np.arange(0,T,dt) # [s]
    len_t = int(len(t))
    np.random.seed(42)
    x = np.random.uniform(-np.sqrt(3), np.sqrt(3), len_t) # uniform distribution over +/-np.sqrt(3), same distribution every time
    #x = np.full_like(t, np.sqrt(3)) # positive stimulus
    #x = np.full_like(t, -np.sqrt(3)) # negative stimulus
    #x = np.zeros_like(t) # zero stimulus

    cell_type = ['on'] * (num_neurons // 2) + ['off'] * (num_neurons // 2)

    run_specifics = {
        'dt': dt,
        't': t,
        'num_neurons': num_neurons,
        'cell_type': cell_type,
        'dc_firing': dc_firing
    }

    print('Computing the LNP Model')
    print(f'dt = {dt}')
    print(f'T = {T}')
    print(f'num_neurons = {num_neurons}')
    print(f'DC_firing = {dc_firing}')
    r, lambda_, u = compute_r(run_specifics, x, plot=plot)
    print('Computed!')

    if save:
        filename = f'lnp_cell{num_neurons}_dt{dt}_T{T}_dc{dc_firing}.npz'
        npz_save(run_specifics, t, x, r, lambda_, u, filename)
        run_specifics, t, x, r, lambda_, u = npz_load(filename)
        print(type(run_specifics))

    return run_specifics, t, x, r, lambda_, u

#run_lnp(0.01, 0.5, 4, 1, save=True, plot=True)