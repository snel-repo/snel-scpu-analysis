"""
PURPOSE: Merge chopped torch outputs with original dataset (and kinematics if available)

REQUIREMENTS: lfads-torch model outputs, original dataset, interface object
                                                ^                 ^
                                                |_________________|
                                        (created in setup_lfads_datasets.py)

OUTPUTS: merged dataset object with original dataset and lfads outputs
"""

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
import yaml
import dill
from analysis_utils import *

# %%
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# load YAML file
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
path_config, ld_cfg, merge_config = load_cfgs(yaml_config_path)


# system inputs
run_date = path_config["RUN_DATE"] # 240108 first run, 240112 second run
expt_name = ld_cfg["NAME"] # Ex: "NP_AAV6-2_ReaChR_184500"
initials = path_config["INITIALS"] # Ex: "cw"
run_type = path_config["TYPE"] # Ex: "spikes"
chan_select = ld_cfg["ARRAY_SELECT"] # Ex: "ALL"
bin_size = ld_cfg["BIN_SIZE"] # Ex: 2

# create paths
ds_name = f"{expt_name}_{chan_select}_{run_type}_{str(bin_size)}"
base_name = f"binsize_{ld_cfg['BIN_SIZE']}"
run_base_dir = f"/snel/share/runs/aav_{run_type}/lfads_{ds_name}/{run_date}_aav_{run_type}_PBT_{initials}"
run_dir = os.path.join(run_base_dir,"best_model")
lfads_torch_outputs_path = os.path.join(run_dir,f"lfads_output_lfads_{ds_name}_out.h5")
lfads_save_dir = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/"
unchopped_ds_path = os.path.join(lfads_save_dir,"lfads_"+ds_name+"_unchopped.pkl")
interface_path = f"{lfads_save_dir}pkls/{ds_name}_interface.pkl"
DATA_FILE = os.path.join(lfads_save_dir, ds_name)

cache_dataset = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/pkls/lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_fulldataset.pkl"

original_h5 = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}.h5"


# %% LOAD CONTINUOUS DATA DF, MERGE WITH TORCH OUTPUTS  
with open(interface_path,'rb') as inf:
    interface = pkl.load(inf)

interface.merge_fields_map = merge_config

with open(cache_dataset,'rb') as inf:
    dataset = pkl.load(inf)

torch_outputs = h5py.File(lfads_torch_outputs_path)

# %% Load chop indices pertaining to training and validation; add to torch output obj if not present

train_inds, valid_inds = get_train_valid_inds(original_h5, torch_outputs, lfads_torch_outputs_path)

# %% Make full output df

data_dict = combine_train_valid_outputs(torch_outputs, train_inds, valid_inds, merge_config)
merged_df = interface.merge(data_dict, smooth_pwr=1)

# %% Merge with original dataset
merge_with_original_df(merged_df, dataset)
# %% smooth spikes, rates, factors
# fill na with 0 for lfads outputs due to chopping
dataset.smooth_spk(gauss_width = 15, name='smooth_15', overwrite=False)

dataset.smooth_spk(signal_type='lfads_rates', gauss_width=8, name='smooth_8', overwrite=False)
dataset.data.lfads_rates_smooth_8 = dataset.data.lfads_rates_smooth_8.fillna(0)

dataset.smooth_spk(signal_type='lfads_factors', gauss_width=8, name='smooth_15', overwrite=False)
dataset.data.lfads_factors_smooth_15 = dataset.data.lfads_factors_smooth_15.fillna(0)

# %%
# save dataset to pickle
merged_full_output = os.path.join(run_dir, f"lfads_{expt_name}_{chan_select}_{run_type}_{bin_size}_full_merged_output.pkl")
with open(merged_full_output, "wb") as f:
    dill.dump(dataset, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

# %%
