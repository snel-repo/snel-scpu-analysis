# %% INPUTS AND PATHS
import os
import pickle as pkl
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from snel_toolkit.datasets.nwb import NWBDataset
import logging
import sys

# %%
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# system inputs
run_date = '240112' # 240108 first run, 240112 second run
expt_name = "NP_AAV6-2_ReaChR_184500"
initials = "cw"
run_type = "spikes" # "spikes" or "emg"
run_name_modifier = "ALL"
samp_rate = 2

# create paths
ds_name = f"{expt_name}_{run_name_modifier}_{run_type}_{str(samp_rate)}"
base_name = "binsize_2ms"
run_base_dir = f"/snel/share/runs/aav_{run_type}/lfads_{ds_name}/{run_date}_aav_{run_type}_PBT_{initials}"
run_dir = os.path.join(run_base_dir,"best_model")
lfads_torch_outputs = os.path.join(run_dir,f"lfads_output_lfads_{ds_name}_out.h5")
lfads_save_dir = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/"
unchopped_ds_path = os.path.join(lfads_save_dir,"lfads_"+ds_name+"_unchopped.pkl")
interface_path = f"{lfads_save_dir}pkls/{ds_name}_interface.pkl"
DATA_FILE = os.path.join(lfads_save_dir, ds_name)

cache_dataset = "/snel/share/share/derived/scpu_snel/nwb_lfads/runs/binsize_2ms/NP_AAV6-2_ReaChR_184500/datasets/lfads_NP_AAV6-2_ReaChR_184500_ALL_spikes_2_fulldataset.pkl"

original_h5 = "/snel/share/share/derived/scpu_snel/nwb_lfads/runs/binsize_2ms/NP_AAV6-2_ReaChR_184500/datasets/lfads_NP_AAV6-2_ReaChR_184500_ALL_spikes_2.h5"


# %% LOAD CONTINUOUS DATA DF, MERGE WITH TORCH OUTPUTS  
with open(interface_path,'rb') as inf:
    interface = pkl.load(inf)


DEFAULT_MERGE_MAP = {
    "output_params": "lfads_rates",
    # "factors": "lfads_factors",
    # "gen_inputs": "lfads_gen_inputs",
}

interface.merge_fields_map = DEFAULT_MERGE_MAP

with open(cache_dataset,'rb') as inf:
    dataset = pkl.load(inf)

torch_outputs = h5py.File(lfads_torch_outputs)

# %% 
def print_spikes(dataset_us):

    base_dir = "/snel/share/share/derived/scpu_snel/NWB/"
    ds_name = "NP_AAV6-2_ReaChR_184500_kilosort.nwb"

    ds_path = os.path.join(base_dir, ds_name)
    logger.info(f"Loading {ds_name} from NWB")
    ds = NWBDataset(ds_path, split_heldout=False)
    BIN_WIDTH = 2 # ms
    ds.resample(BIN_WIDTH)

    print(np.array_equal(dataset_us.data.spikes.values, ds.data.spikes.values))
    print("our dataset shape", dataset_us.data.spikes.values.shape)
    print("nwb dataset shape", ds.data.spikes.values.shape)

    # print number of indices that differ between the datasets
    print(np.sum(np.abs(dataset_us.data.spikes.values - ds.data.spikes.values)))

    # print which columns have differences
    print(np.where(np.abs(dataset_us.data.spikes.values - ds.data.spikes.values) > 0))

# print_spikes(dataset)

# %% 
original_h5_data = h5py.File(original_h5)
train_inds = original_h5_data['train_inds'][()]
valid_inds = original_h5_data['valid_inds'][()]

# %% RUN ONCE
# with h5py.File(lfads_torch_outputs,'a') as torch_output_data:
#     torch_output_data.create_dataset('train_inds',data=train_inds)
#     torch_output_data.create_dataset('valid_inds',data=valid_inds)


# %% Make full output df
    
n_batch = train_inds.size + valid_inds.size
train_output = torch_outputs['train_output_params'][()]
valid_output = torch_outputs['valid_output_params'][()]
full_output = np.empty((n_batch, train_output.shape[1], train_output.shape[2]))
full_output[train_inds,:,:] = train_output
full_output[valid_inds,:,:] = valid_output

data_dict = {}
data_dict['output_params'] = full_output

merged_df = interface.merge(data_dict, smooth_pwr=1)

# %% Merge with original dataset
chan_names=dataset.data['spikes'].columns.values
dataset.add_continuous_data(
    merged_df['lfads_rates'].values,
    "lfads_rates",
    chan_names=chan_names,
)
# %% smooth spikes
dataset.smooth_spk(gauss_width = 15, name='smooth_15', overwrite=False)

# %% all channels raster for single trial
event_id = 25 # locomotion/stim trial idx
# LOCOMOTION: 1, 2*, 4* good; 0, 3 okay | 
# STIM: all but 19  | 18, 23, 24, 25/26/27 (periodic after) good examples
BIN_WIDTH = 2 # ms


win_len_ms = 4000 # ms
pre_buffer_ms = 500 # ms
win_len = win_len_ms / BIN_WIDTH
pre_buff = pre_buffer_ms / BIN_WIDTH
event_start_time = dataset.trial_info.iloc[event_id].start_time - pd.to_timedelta(pre_buffer_ms, unit="ms")
event_type = dataset.trial_info.iloc[event_id].event_type
start_ix = dataset.data.index.get_loc(event_start_time, method='nearest')
stop_ix = int(start_ix + win_len)
# make a 2 by 1 subplot
fig,axs = plt.subplots(3,1,figsize=(8,11),dpi=100)
plt.subplots_adjust(top=0.88)

# fig,axs = plt.subplot(2,1,figsize=(10,4),dpi=100)
# fig = plt.figure(figsize=(10,4), dpi=100)
smooth_slice = dataset.data.spikes_smooth_15.values[start_ix:stop_ix,:]
lfads_slice = dataset.data.lfads_rates.values[start_ix:stop_ix,:]
spikes_slice = dataset.data.spikes.values[start_ix:stop_ix,:]
vmin = 0 # min([smooth_slice.min(), lfads_slice.min()])
vmax = max([smooth_slice.max(), lfads_slice.max()])
max_scale = 0.5
time_vec = dataset.data.index.values[start_ix:stop_ix+1].astype("timedelta64[ms]").astype(float)/1000.0

fig.suptitle(f"{event_type} trial {event_id}", x=0.1,y=1)
axs[0].pcolor(smooth_slice.T, cmap='viridis', vmin=vmin, vmax=max_scale*vmax)
axs[1].pcolor(lfads_slice.T, cmap='viridis', vmin=vmin, vmax=max_scale*vmax)
axs[2].pcolor(spikes_slice.T, cmap=colormap.bone_r, vmin=0, vmax=1)

axs[0].set_title("smoothed spikes")
axs[0].vlines(pre_buff, 0, dataset.data.spikes.shape[1], color="r")
axs[0].set_xticklabels(time_vec[axs[0].get_xticks().astype(int)])
axs[0].set_ylabel("neurons")
axs[0].set_xlabel("time (s)")
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)

axs[1].set_title("LFADS inferred rates")
axs[1].vlines(pre_buff, 0, dataset.data.spikes.shape[1], color="r")
axs[1].set_xticklabels(time_vec[axs[1].get_xticks().astype(int)])
axs[1].set_ylabel("neurons")
axs[1].set_xlabel("time (s)")
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

axs[2].set_title("original spikes")
axs[2].vlines(pre_buff, 0, dataset.data.spikes.shape[1], color="r")
axs[2].set_xticklabels(time_vec[axs[2].get_xticks().astype(int)])
axs[2].set_ylabel("neurons")
axs[2].set_xlabel("time (s)")
axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)

plt.subplots_adjust(hspace=0.5)
fig.tight_layout()
plt.show()

# %%
chans = [34, 68, 85, 47, 79]
# start_chan_ix = 34

pre_offset_ms = -100
post_offset_ms = 200
stim_line_x = -1*(pre_offset_ms / BIN_WIDTH)
time_vec = np.arange(pre_offset_ms, post_offset_ms, BIN_WIDTH)
events = dataset.trial_info[dataset.trial_info.event_type == "stimulation"].trial_id.values

fig, axs = plt.subplots(2, len(chans), figsize=(10,3.5), dpi=150)
axs = axs.flatten()

for i, chan in enumerate(chans):
    trial_dat_spikes = []
    trial_dat_lfads = []
    if dataset.data.spikes.columns.isin([chan]).any():
        for event in events:        
            event_start_time = dataset.trial_info.iloc[event].start_time + pd.to_timedelta(pre_offset_ms, unit="ms")    
            event_stop_time = dataset.trial_info.iloc[event].start_time + pd.to_timedelta(post_offset_ms, unit="ms")    
            start_ix = dataset.data.index.get_loc(event_start_time, method='nearest')
            stop_ix = dataset.data.index.get_loc(event_stop_time, method='nearest')
            dat_spikes = dataset.data.spikes.iloc[start_ix:stop_ix, chan].values
            trial_dat_spikes.append(dat_spikes)


            dat_lfads = dataset.data.lfads_rates.iloc[start_ix:stop_ix, chan].values
            trial_dat_lfads.append(dat_lfads)
        axs[i].pcolor(np.array(trial_dat_spikes), cmap=colormap.bone_r, vmin=0, vmax=2)    
        axs[i].set_title(f"channel {chan}", fontsize=8)
        axs[i].vlines(stim_line_x, 0, len(trial_dat_lfads), color="r")
        xticks = axs[i].get_xticks().astype(int)
        xticks = xticks*BIN_WIDTH + pre_offset_ms
        axs[i].set_xticklabels(xticks)
        #ax.set_ylabel("trials")
        #ax.set_xlabel("time (s)")
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i+len(chans)].pcolor(np.array(trial_dat_lfads), cmap='viridis', vmin=0, vmax=np.max(np.array(trial_dat_lfads)))    
        axs[i+len(chans)].set_title(f"channel {chan}", fontsize=8)
        axs[i+len(chans)].vlines(stim_line_x, 0, len(trial_dat_lfads), color="r")
        xticks = axs[i+len(chans)].get_xticks().astype(int)
        xticks = xticks*BIN_WIDTH + pre_offset_ms
        axs[i+len(chans)].set_xticklabels(xticks)
        #ax.set_ylabel("trials")
        #ax.set_xlabel("time (s)")
        axs[i+len(chans)].spines['right'].set_visible(False)
        axs[i+len(chans)].spines['top'].set_visible(False)
    plt.plot()
fig.tight_layout()

# %% SAME AS ABOVE BUT PLOTS MULTIPLE

start_chan_ix = 75
plot_n_chans = 9

width = np.ceil(np.sqrt(plot_n_chans)).astype(int)
height = np.floor(np.sqrt(plot_n_chans)).astype(int)
pre_offset_ms = -100
post_offset_ms = 200
stim_line_x = -1*(pre_offset_ms / BIN_WIDTH)
time_vec = np.arange(pre_offset_ms, post_offset_ms, BIN_WIDTH)
events = dataset.trial_info[dataset.trial_info.event_type == "stimulation"].trial_id.values
fig, axs = plt.subplots(width, height, figsize=(6,10), dpi=150)
for i in range(plot_n_chans):
    axs = axs.flatten()
    trial_dat_spikes = []
    trial_dat_lfads = []
    chan_ix = start_chan_ix + i
    if dataset.data.spikes.columns.isin([chan_ix]).any():
        for event in events:        
            event_start_time = dataset.trial_info.iloc[event].start_time + pd.to_timedelta(pre_offset_ms, unit="ms")    
            event_stop_time = dataset.trial_info.iloc[event].start_time + pd.to_timedelta(post_offset_ms, unit="ms")    
            start_ix = dataset.data.index.get_loc(event_start_time, method='nearest')
            stop_ix = dataset.data.index.get_loc(event_stop_time, method='nearest')
            dat_spikes = dataset.data.spikes.iloc[start_ix:stop_ix, chan_ix].values
            trial_dat_spikes.append(dat_spikes)

            dat_lfads = dataset.data.lfads_rates.iloc[start_ix:stop_ix, chan_ix].values
            trial_dat_lfads.append(dat_lfads)
        axs[0].pcolor(np.array(trial_dat_spikes), cmap=colormap.bone_r, vmin=0, vmax=2)    
        axs[0].set_title(f"channel {chan_ix}", fontsize=8)
        axs[0].vlines(stim_line_x, 0, len(trial_dat_lfads), color="r")
        xticks = axs[0].get_xticks().astype(int)
        xticks = xticks*BIN_WIDTH + pre_offset_ms
        axs[0].set_xticklabels(xticks)
        #ax.set_ylabel("trials")
        #ax.set_xlabel("time (s)")
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        axs[1].pcolor(np.array(trial_dat_lfads), cmap=colormap.bone_r, vmin=0, vmax=2)    
        axs[1].set_title(f"channel {chan_ix}", fontsize=8)
        axs[1].vlines(stim_line_x, 0, len(trial_dat_lfads), color="r")
        xticks = axs[1].get_xticks().astype(int)
        xticks = xticks*BIN_WIDTH + pre_offset_ms
        axs[1].set_xticklabels(xticks)
        #ax.set_ylabel("trials")
        #ax.set_xlabel("time (s)")
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
fig.tight_layout()
