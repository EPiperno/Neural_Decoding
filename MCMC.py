import numpy as np
import os
import matplotlib.pyplot as plt
import LNP_model as lnp
import MAP as map
import pickle
import time

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'Data/')
figure_path = os.path.join(current_path, 'Figures/')

def compute_log_pi_x(run_specifics, r, x_sample):
    log_p_rx = map.compute_log_p_rx(run_specifics, r, x_sample)
    return log_p_rx

def compute_x_MH(run_specifics, r, M, B):
    # returns x_MH, which are  N = M-B samples of the stimulus to approximate the posterior mean
    # It is an efficient way to approximate the computationally expensive posterior mean, by taking samples with high probability
    # M: number of samples to generate
    # B: number of burn-in samples
    # x_MH: N x len(t) array of samples

    dt, t, num_neurons, cell_type, dc_firing = lnp.unravel_run_specifics(run_specifics)
    x_MH = np.zeros((M-B, len(t)))
    x_t = np.random.uniform(-np.sqrt(3), np.sqrt(3), len(t))
    for i in range(M):
        # x_t = N MH i.i.d. samples from the distribution pi_x
        print(f'Iteration {i}')
        if i == 0:
            x_t = np.random.uniform(-np.sqrt(3), np.sqrt(3), len(t))
        else:
            x_t_candidate = np.clip(x_t + np.random.normal(0, 0.1, len(t)), -np.sqrt(3), np.sqrt(3))
            log_pi_x_t = compute_log_pi_x(run_specifics, r, x_t)
            log_pi_x_t_candidate = compute_log_pi_x(run_specifics, r, x_t_candidate)
            log_acceptance_ratio = log_pi_x_t_candidate - log_pi_x_t
            proposal_ratio = np.prod(np.clip(x_t_candidate + np.random.normal(0, 0.1, len(t)), -np.sqrt(3), np.sqrt(3)) / x_t)
            acceptance_ratio = np.exp(log_acceptance_ratio) * proposal_ratio

            if np.random.rand() < acceptance_ratio:
                x_t = x_t_candidate
                print(f'log_pi_x_t_candidate = {log_pi_x_t_candidate}')

        if i >= B:
            x_MH[i-B] = x_t
    

    return x_MH

def compute_x_BLS(run_specifics, r, M=1000, B=100, save=True):
    # returns x_BLS, the posterior mean of the stimulus given the spike train
    x_MH = compute_x_MH(run_specifics, r, M, B)
    x_BLS = np.mean(x_MH, axis=0)

    if save:
        filename = f'x_BLS_M{M}_B{B}.npz'
        output_path = os.path.join(data_path, filename)
        np.savez(output_path, x_BLS=x_BLS)

        loaded_data = np.load(output_path)
        x_BLS_loaded = loaded_data['x_BLS']
        print(f'Loaded x_BLS: {x_BLS_loaded}')
    return x_BLS

def load_x_BLS(M, B):
    filename = f'x_BLS_M{M}_B{B}.npz'
    output_path = os.path.join(data_path, filename)
    loaded_data = np.load(output_path)
    x_BLS = loaded_data['x_BLS']
    return x_BLS

def run_MCMC(dt, T, num_neurons, dc_firing):
    lnp_filename = f'lnp_cell{num_neurons}_dt{dt}_T{T}_dc{dc_firing}'

    try:
        run_specifics, t, x, r, lambda_, u = lnp.npz_load(lnp_filename + '.npz')
    except:
        run_specifics, t, x, r, lambda_, u = lnp.run_lnp(dt, T, num_neurons, dc_firing, save=True, plot=True)
    
    run_specifics = {
        'dt': dt,
        't': np.arange(0,T,dt),
        'num_neurons': num_neurons,
        'cell_type': ['on'] * (num_neurons // 2) + ['off'] * (num_neurons // 2),
        'dc_firing': dc_firing
        }
    
    log_pi_x_og = compute_log_pi_x(run_specifics, r, x)
    print(f'log_pi_x_og = {log_pi_x_og}')

    M_arr = [10]#, 20, 50, 100, 200, 500]
    B_arr = [1]#, 2, 5, 10, 20, 50]
    time_arr = []
    x_BLS_arr = []
    msq_error_arr = []

    for i, M in enumerate(M_arr):
        start_time = time.time()
        x_BLS = compute_x_BLS(run_specifics, r, M=M_arr[i], B=B_arr[i])
        end_time = time.time()
        time_arr.append(end_time-start_time)
        x_BLS_arr.append(x_BLS)
        msq_error_arr.append(lnp.compute_msq_error(x, x_BLS))

    print(f'M_arr = {M_arr}')
    print(f'B_arr = {B_arr}')
    print(f'time_arr = {time_arr}')
    print(f'msq_error_arr = {msq_error_arr}')

    mcmc_filename = 'mcmc_M_10'

    full_filename = lnp_filename + mcmc_filename + '.npz'
    output_path = os.path.join(data_path, full_filename)

    results = {
        'M_arr': M_arr,
        'B_arr': B_arr,
        'time_arr': time_arr,
        'x_BLS_arr': x_BLS_arr,
        'msq_error_arr': msq_error_arr
    }

    np.savez(output_path, **results)

    loaded_results = np.load(output_path)
    print(f'Loaded results: {loaded_results}')

#run_MCMC(0.01, 0.5, 4, 1)