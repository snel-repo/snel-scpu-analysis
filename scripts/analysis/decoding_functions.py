
import numpy as np
import typing
from sklearn.linear_model import Ridge
import typing
from typing import List
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
SAMPLE_RATE = 500 # hertz
"""
This function takes a numpy array and returns concatenated data based on start and stop indices
"""
def concat_data_given_start_stop_indices(dataset: np.ndarray, start_indices: List[int], stop_indices: List[int]) -> np.ndarray:
    assert len(start_indices) == len(stop_indices), "Start and stop indices lists must be of the same length"
    
    concatenated_data = np.concatenate(tuple(dataset[start:stop] for start, stop in zip(start_indices, stop_indices)), axis=0)
    
    return concatenated_data



"""
Returns all nonNan data of kinematic data for all given body parts and returns lfads rates based on those indices as well as the column names

"""
def return_all_nonNan_slice(dataset, use_smooth_data: bool = True, use_LFADS: bool = True) -> typing.Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if use_smooth_data:
        all_kin_data = dataset.data.kin_pos_smooth_3.copy()
        if use_LFADS:
            all_rates_data = dataset.data.lfads_rates_smooth_8
        else:
            all_rates_data = dataset.data.spikes_smooth_25
    else:
        all_kin_data = dataset.data.kin_pos.copy()
        if use_LFADS:
            all_rates_data = dataset.data.lfads_rates
        else:
            all_rates_data = dataset.data.spikes_smooth_25

    kin_column_names = all_kin_data.columns
    all_kin_data['Original_Index'] = np.arange(len(all_kin_data))
    all_kin_data_nonNan = all_kin_data.dropna().copy()
    non_Nan_indices = all_kin_data_nonNan['Original_Index'].values
    rates_slice = all_rates_data.iloc[non_Nan_indices]
    return all_kin_data_nonNan, rates_slice, kin_column_names


def diff_filter(x):
        """differentation filter"""
        return signal.savgol_filter(x, 27, 5, deriv=1, axis=0)


def linear_regression_train_val(x, y, alpha=0, folds=5):
    """
    Function to perform linear regression with cross validation to decode kinematic data from neural data (eg. lfads rates, factors, smooth spikes)
    x is 2D array of shape (n_bins, n_channels) which is (samples, features)
    y is 1D array of shape (n_bins,) where each value is kinematic data
    """
    from sklearn.preprocessing import StandardScaler
    kf = KFold(n_splits=folds,  shuffle=True, random_state=42)
    lr = Ridge(alpha=alpha)

    r2_test = np.zeros(folds)
    r2_train = np.zeros(folds)

    test_pred = np.zeros_like(y)
    scaler = StandardScaler()
    for split_num, (train_ix, test_ix) in enumerate(kf.split(x)):
        x_train, x_test = x[train_ix, :], x[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        # normalize predictor
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # normalize target
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = scaler.transform(y_test.reshape(-1, 1)).ravel()

        lr.fit(x_train, y_train)
        test_pred[test_ix] = lr.predict(x_test)
        train_pred = lr.predict(x_train)
        r2_test[split_num] = r2_score(y_test, test_pred[test_ix])
        r2_train[split_num] = r2_score(y_train, train_pred)

    
    test_sem = np.std(r2_test) / np.sqrt(folds)
    train_sem = np.std(r2_train) / np.sqrt(folds)
 
    r2_test = np.mean(r2_test)
    r2_train = np.mean(r2_train)


    return test_pred, r2_test, r2_train, train_sem, test_sem

def append_kinematic_angle_data(all_kin_df: pd.DataFrame) -> pd.DataFrame:
    """
        Calculates and Appends knee and ankle angle data to kinematic data and returns all kin data and angles in one dataframe
        Return angles are in degrees
    """
    # get knee and ankle angle data
    ankle_kinematics_np = all_kin_df[["ankle_x", "ankle_y"]].values
    hip_kinematics_np = all_kin_df[["hip_x", "hip_y"]].values
    iliac_crest_kinematics_np = all_kin_df[["iliac_crest_x", "iliac_crest_y"]].values
    knee_kinematics_np = all_kin_df[["knee_x", "knee_y"]].values
    toe_kinematics_np = all_kin_df[["toe_x", "toe_y"]].values

    hip_to_knee_vectors = knee_kinematics_np - hip_kinematics_np
    knee_to_ankle_vectors = ankle_kinematics_np - knee_kinematics_np
    ankle_to_toe_vectors = toe_kinematics_np - ankle_kinematics_np

    # Dot product and magnitudes for knee angle
    dot_product_knee = np.sum(hip_to_knee_vectors * knee_to_ankle_vectors, axis=1)
    mag_hip_to_knee = np.linalg.norm(hip_to_knee_vectors, axis=1)
    mag_knee_to_ankle = np.linalg.norm(knee_to_ankle_vectors, axis=1)

    # Dot product and magnitudes for ankle angle
    dot_product_ankle = np.sum(knee_to_ankle_vectors * ankle_to_toe_vectors, axis=1)
    mag_ankle_to_toe = np.linalg.norm(ankle_to_toe_vectors, axis=1)

    # Angle calculation
    knee_angles_vectorized = np.arccos(dot_product_knee / (mag_hip_to_knee * mag_knee_to_ankle)) 
    ankle_angles_vectorized = np.arccos(dot_product_ankle / (mag_knee_to_ankle * mag_ankle_to_toe))

    # phase unwrapping for knee and ankle angles
    knee_angles_vectorized = np.unwrap(knee_angles_vectorized)
    ankle_angles_vectorized = np.unwrap(ankle_angles_vectorized)

    # convert to degrees
    knee_angles_vectorized = np.degrees(knee_angles_vectorized)
    ankle_angles_vectorized = np.degrees(ankle_angles_vectorized)

    
    all_kin_df["knee_angle"] = knee_angles_vectorized
    all_kin_df["ankle_angle"] = ankle_angles_vectorized

    return all_kin_df


def plot_rates_vel_reg(regression_vel_slice, regression_rates_slice, orig_indices):
    """
    Plots velocity and rates (input to linear regression model)
    """

    assert len(orig_indices) == len(regression_vel_slice), "Length of original indices must be the same as the length of the data"
    assert len(orig_indices) == regression_rates_slice.shape[0], "Length of original indices must be the same as the length of the data"
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    
    fig.suptitle(f'Velocity and Rates indices {orig_indices[0], orig_indices[-1]}')
    fig.subplots_adjust(hspace=0.25)

    ax[0].plot(regression_vel_slice)
    ax[0].set_title('Velocity')
    ax[0].set_xlabel('Time (bins)')
    ax[0].set_ylabel('Velocity')
    vmax = np.max(regression_rates_slice)
    c = ax[1].pcolor(regression_rates_slice.T, cmap='viridis', vmin=0, vmax=vmax)
    ax[1].set_title('Rates')
    ax[1].set_xlabel('Time (bins)')
    ax[1].set_ylabel('Channels')
    plt.colorbar(c)
    plt.show()

def plot_psd_kinematic_data(kin_slice, use_smooth_data, bin_size=2):
    """
    Plots power spectral density of kinematic data
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fs = 1/(bin_size/1000) # sampling frequency in Hz
    f, Pxx_den = signal.welch(kin_slice, fs, nperseg=256)
    ax.semilogy(f, Pxx_den)
    title = f'PSD: kinematic data Smooth: cutoff: {cutoff_freq}' if use_smooth_data else f'PSD: kinematic data'
    ax.set_title(title)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    plt.show()

def preprocessing(dataset, column_name, use_smooth_data, start_indices, stop_indices, plot=False, use_LFADS=True, use_velocity=True):
    """
        preprocessing steps. output will be used for linear regression
    """
    print('beginning preprocessing')
    # PREPROCESSING
    # all_kin_df has a new column called original indices which are the nonNan indices
    all_kin_df, rates_slice, _ = return_all_nonNan_slice(dataset, use_smooth_data=use_smooth_data, use_LFADS=use_LFADS)
   
    all_kin_slice_and_angle_df = append_kinematic_angle_data(all_kin_df)

    regression_vel_slice = all_kin_slice_and_angle_df[column_name].values

   

    cutoff_freq = 5
    nyquist = 0.5 * SAMPLE_RATE
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(2, normal_cutoff, btype='low', analog=False)

    if use_velocity:
        regression_vel_slice = diff_filter(regression_vel_slice)

    regression_vel_slice = signal.filtfilt(b, a, regression_vel_slice)



    regression_rates_slice = np.log(rates_slice + 1e-10)

    # subselect data from non NaN indices
    assert len(start_indices) == len(stop_indices)
    if len(start_indices) >= 1: # if start and stop indices are provided

        # subset indices are the indices of nonNan data, so we can use it to get the original indices

        regression_vel_slice = concat_data_given_start_stop_indices(regression_vel_slice, start_indices, stop_indices)
        
        regression_rates_slice = concat_data_given_start_stop_indices(regression_rates_slice, start_indices, stop_indices)
        
        orig_indices = concat_data_given_start_stop_indices(all_kin_df['Original_Index'].values, start_indices, stop_indices)
    else:
        orig_indices = all_kin_df['Original_Index'].values
    # index_smallest_10 = np.argpartition(regression_vel_slice, 10)[:10]
    # index_largest_10 = np.argpartition(regression_vel_slice, -10)[-10:]
    # outlier_indices = np.concatenate((index_smallest_10, index_largest_10))
    
    # outlier_min = np.argmin(regression_vel_slice)
    # val_min = regression_vel_slice[outlier_min]
    # print(val_min)
    # outlier_max = np.argmax(regression_vel_slice)
    # regression_vel_slice = np.delete(regression_vel_slice, outlier_indices)
    # regression_rates_slice = np.delete(regression_rates_slice, outlier_indices, axis=0)

    # apply wiener filter
    # m = 20
    # regression_vel_slice = np.convolve(regression_vel_slice, np.ones((m,))/m, mode='same')

    # delete channels with large variance
    # var = np.var(regression_rates_slice, axis=0)
    # large_variance_channels = np.where(var > 5)
    # regression_rates_slice = np.delete(regression_rates_slice, large_variance_channels, axis=1) # remove channels with large variance
    if plot:
        plot_rates_slice = np.exp(regression_rates_slice) - 1e-10
        plot_rates_vel_reg(regression_vel_slice, plot_rates_slice, orig_indices)
        #plot_psd_kinematic_data(regression_vel_slice, use_smooth_data, bin_size=2)
    return regression_vel_slice, regression_rates_slice, orig_indices
