import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'Data/')
figure_path = os.path.join(current_path, 'Figures/')

def extract_filter(cell_type, filter_type, time_array):

    if cell_type == 'off':
        if filter_type == 'k':
            filename = 'off_cell_stimulus_filter.csv'
        elif filter_type == 'h':
            filename = 'off_cell_postspike_filter.csv'
    elif cell_type == 'on':
        if filter_type == 'k':
            filename = 'on_cell_stimulus_filter.csv'
        elif filter_type == 'h':
            filename ='on_cell_postspike_filter.csv'

    file_path = os.path.join(data_path, filename)
    pd_data = pd.read_csv(file_path)
    t = pd_data.iloc[:, 0]
    y = pd_data.iloc[:, 1]

    y_interpolated = np.interp(time_array, t, y)

    return y_interpolated

def extract_filter_t(cell_type, filter_type, t_i):
    if cell_type == 'off':
        if filter_type == 'k':
            filename = 'off_cell_stimulus_filter.csv'
        elif filter_type == 'h':
            filename = 'off_cell_postspike_filter.csv'
    elif cell_type == 'on':
        if filter_type == 'k':
            filename = 'on_cell_stimulus_filter.csv'
        elif filter_type == 'h':
            filename ='on_cell_postspike_filter.csv'

    file_path = os.path.join(data_path, filename)
    pd_data = pd.read_csv(file_path)
    t = pd_data.iloc[:, 0]
    y = pd_data.iloc[:, 1]

    y_extrapolated = np.interp([t_i], t, y)[0]

    return y_extrapolated

# # Stimulus Filter Figure
# t = np.arange(-0.25,0,0.001) # [s]
# k_on = extract_filter('on', 'k', t)
# k_off = extract_filter('off', 'k', t)

# plt.figure(figsize=(7.5,5))  
# plt.plot(t, k_on, label = 'ON Cell')
# plt.plot(t, k_off, label = 'OFF Cell')
# plt.xlabel('Time [s]')
# plt.ylabel('Stimulus Filter Value [a.u.]')
# plt.legend()
# plt.show()