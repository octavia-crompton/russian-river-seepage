from itertools import groupby

import numpy as np
import pandas as pd

def sequence_lengths(data):
    """
    identify lengths of inlet closures
    """
    lengths = []
    for key, group in groupby(data):
        group_list = list(group)
        if key == 1:
            lengths.extend([len(group_list)] * len(group_list))
        else:
            lengths.extend([0] * len(group_list))
    return lengths

# ###### ###### ###### ###### ###### ###### #####

# Functions to compute the Short Time Fourier Transform (STFT)
import numpy as np
import pandas as pd
from scipy.signal import stft

def add_power_USGS(subset, period, data_freq='15T', label='USGS_24_hr'):
    """
    Add power at user-specified period for different data frequencies.
    
    Input:
        subset: dataset to modify, frequency should be either '15T' for 15-minute or 'H' for hourly.
        period: in hours
        data_freq: data frequency, '15T' for 15 minutes or 'H' for hourly
        label: label for new column

    Note: returns power^(1/4)
    """
    if data_freq == '15T':
        samples_per_hour = 4
    elif data_freq == 'H':
        samples_per_hour = 1
    else:
        raise ValueError("data_freq must be '15T' for 15-minute or 'H' for hourly data")
    
    # Calculate the sampling frequency and nperseg for STFT
    fs = samples_per_hour / 3600  # samples per second
    nperseg = 4*samples_per_hour * period
    
    # Apply STFT Short Time Fourier Transform (STFT)
    f, t, Zxx = stft(subset['USGS_filled'], fs=fs, nperseg=nperseg, noverlap=nperseg-1) 

    # Find the index corresponding to the period-hour frequency
    freq_index = np.argmin(np.abs(f - 1/(period*3600)))
    
    # Extract the power at the period-hour frequency
    USGS_period_hr = np.abs(Zxx[freq_index, :])**0.25
    
    # Convert the time array from seconds to timedelta, then to the original timestamp
    time_deltas = pd.to_timedelta(t, unit='s')
    time_stamps = subset.index[0] + time_deltas
    
    # Create a DataFrame for the interpolated power data
    power_subset = pd.DataFrame(data={label: USGS_period_hr}, index=time_stamps)
    
    # Interpolate the power data to match the original DataFrame's time index
    power_subset = power_subset.reindex(power_subset.index.union(subset.index)).interpolate('index').loc[subset.index]
    
    # Add the interpolated power data to the original DataFrame
    subset[label] = power_subset[label]/power_subset[label].max()
    
    return subset



def add_power_visitor(subset, period, data_freq='15T', label = 'visitor_24_hr' ):
    """
    Add power at user-specified period
    
    Input:
        period : in hours
        subset : dataset to modify, frequency should by 15 minute
        label : label new column

    Note: returns power^(1/4)
    """

    if data_freq == '15T':
        samples_per_hour = 4
    elif data_freq == 'H':
        samples_per_hour = 1
    else:
        raise ValueError("data_freq must be '15T' for 15-minute or 'H' for hourly data")
    
    # Calculate the sampling frequency and nperseg for STFT
    fs = samples_per_hour / 3600  # samples per second
    nperseg = 4*samples_per_hour * period
    
    # Apply STFT Short Time Fourier Transform (STFT)
    f, t, Zxx = stft(subset['visitor_filled'], fs=fs, nperseg=nperseg, noverlap=nperseg-1) 
    # Apply STFT  Short Time Fourier Transform (STFT)
    #f, t, Zxx = stft(subset['visitor_filled'],fs=1/(3600), nperseg=4*period, noverlap=4*period-1) 

    # Find the index corresponding to the period-hour frequency
    # f is an array of frequencies. 1/(period*3600) is the period-hour frequency in Hz
    freq_index = np.argmin(np.abs(f - 1/(period*3600)))
    # Extract the power at the period-hour frequency
    visitor_period_hr = np.abs(Zxx[freq_index, :])**0.5#**2
    # Convert the time array from seconds to timedelta, then to the original timestamp
    time_deltas = pd.to_timedelta(t, unit='s')
    time_stamps = subset.index[0] + time_deltas

    # Create a DataFrame for the interpolated power data
    power_subset = pd.DataFrame(data={label: visitor_period_hr}, index=time_stamps)
    # Interpolate the power data to match the original DataFrame's time index
    power_subset = power_subset.reindex(power_subset.index.union(subset.index)).interpolate('index').loc[subset.index]

    # Add the interpolated power data to the original DataFrame
    subset[label] = power_subset[label]/power_subset[label].max()

    return subset
