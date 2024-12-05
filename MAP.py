import numpy as np
import os
import matplotlib.pyplot as plt
import LNP_model as lnp
import pickle
from scipy.optimize import minimize
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'Data/')
figure_path = os.path.join(current_path, 'Figures/')

def compute_log_p_rx(run_specifics, r, x_sample):
    dt, t, num_neurons, cell_type, dc_firing = lnp.unravel_run_specifics(run_specifics)
    lambda_ = lnp.compute_lambda_(run_specifics, x_sample)
    log_p_rx = 0
    for i in range(len(r)):
        term1 = np.sum(r[i] * np.log(lambda_[i]))
        try:
            term2 = np.trapezoid(lambda_[i], t)
        except:
            term2 = np.trapz(lambda_[i], t)
        log_p_rx += term1 - term2    
    return log_p_rx

def compute_log_p_rx_fast(t, r, lambda_):
    log_p_rx = 0
    for i in range(len(r)):
        term1 = np.sum(r[i] * np.log(lambda_[i]))
        try:
            term2 = np.trapezoid(lambda_[i], t)
        except:
            term2 = np.trapz(lambda_[i], t)
        log_p_rx += term1 - term2    
    return log_p_rx

def compute_log_p_x(x):
    # We consider a flat distribution of x, thus log(p(x)) is irrelevant in the x_map calculation
    return 0

def run_x_map(dt, T, num_neurons, dc_firing):
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
    
    def compute_max_func(x):
        lambda_ = lnp.compute_lambda_(run_specifics, x)
        max_func = compute_log_p_rx_fast(t, r, lambda_)
        return max_func

    options = {'disp': True}

    def callback(xk):
        print(f"Current x: {xk}")

    class ProgressCallback:
        def __init__(self, total):
            self.pbar = tqdm(total=total)
            self.iteration = 0

        def __call__(self, xk):
            self.iteration += 1
            self.pbar.update(1)
            print(f"Current x: {xk}")

        def close(self):
            self.pbar.close()

    progress_callback = ProgressCallback(total=100)  # Adjust the total as needed

    result = minimize(lambda x: -compute_max_func(x), x0=np.zeros_like(t), bounds=[(-np.sqrt(3), np.sqrt(3))]*len(t), options=options, callback=progress_callback)

    progress_callback.close()
    x_map = result.x
    print(f'x_map: {x_map}')
    
    return x_map

#run_x_map(0.01, 0.5, 4, 1)

