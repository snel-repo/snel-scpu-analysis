'''
PURPOSE: Chops continuous NWB data into overlapping windows for LFADS input

Script resamples data, drops channels, and saves chopped data as h5 file

OUTPUT: 
1. Resampled SNEL Toolkit Dataset object saved as pickle
2. LFADS config file saved as yaml
3. SNEL Toolkit LFADSInterface object saved as pickle
'''
###
# %%
import sys
import os
import h5py
import _pickle as pickle
from snel_toolkit.datasets.nwb import NWBDataset
from snel_toolkit.datasets.base import DataWrangler
from snel_toolkit.interfaces import deEMGInterface, LFADSInterface
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

import logging
import yaml
import numpy as np
import pandas as pd

# %%
# --- setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# === begin SCRIPT PARAMETERS ==========================
# load yaml config file
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
lfads_dataset_cfg = yaml.load(open(yaml_config_path), Loader=yaml.FullLoader)
path_config = lfads_dataset_cfg["PATH_CONFIG"]
ld_cfg = lfads_dataset_cfg["DATASET"]
chop_cfg = lfads_dataset_cfg["CHOP_PARAMETERS"]

expt_name = ld_cfg["NAME"]

# %%
# -- paths
base_name = f"binsize_{ld_cfg['BIN_SIZE']}"
ds_base_dir = "/snel/share/share/derived/scpu_snel/NWB/"
lfads_save_dir = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/"

# === end SCRIPT PARAMETERS ==========================


ds_path = os.path.join(ds_base_dir, expt_name + ".nwb")
# --- load dataset from NWB
logger.info(f"Loading {expt_name} from NWB")
dataset = NWBDataset(ds_path, split_heldout=False)

# --- preprocess spiking data



# --- drop spk channnels (if necessary)

# -- extract relevant config params
ARRAY_SELECT = ld_cfg["ARRAY_SELECT"]
XCORR_THRESHOLD = ld_cfg["SPK_XCORR_THRESHOLD"]

# --- xcorr rejection
# check that analysis is happening at 1ms
assert dataset.bin_width == 1
pair_xcorr, drop_spk_names = dataset.get_pair_xcorr(
    "spikes", threshold=XCORR_THRESHOLD, zero_chans=True
)

# --- resample dataset (if necessary)
# -- extract relevant config params
BIN_SIZE = ld_cfg["BIN_SIZE"]
if dataset.bin_width != BIN_SIZE:
    logger.info(f"Resampling dataset to bin width (ms): {BIN_SIZE}")
    dataset.resample(BIN_SIZE)

chop_df = dataset.data

# -- drop spk channels
logger.info(f"Drop spike channels: {len(drop_spk_names)}/{chop_df.spikes.columns.values.size}")

# if len(drop_spk_names) > 0:
#     chop_df.drop(columns=drop_spk_names, axis=1, level=1, inplace=True)

# --- create save dirs if they do not exist
pkl_dir = os.path.join(lfads_save_dir, "pkls")
if os.path.exists(lfads_save_dir) is not True:
    os.makedirs(lfads_save_dir)
if os.path.exists(pkl_dir) is not True:
    os.makedirs(pkl_dir)

# --- initialize chop interface

# -- extract relevant config params
WIN_LEN = chop_cfg["WINDOW"]
OLAP_LEN = chop_cfg["OVERLAP"]
MAX_OFF = chop_cfg["MAX_OFFSET"]
CHOP_MARG = chop_cfg["CHOP_MARGINS"]
RAND_SEED = chop_cfg["RANDOM_SEED"]
TYPE = path_config["TYPE"]
NAME = ld_cfg["NAME"]


# setup initial chop fields map (defines which fields will be chopped for lfads)
chop_fields_map = {chop_cfg["DATA_FIELDNAME"]: "data"}

# if we are using external inputs, add this field to chop map
if chop_cfg["USE_EXT_INPUT"]:
    logger.info(
        f"Setting up lfads dataset with external inputs from {chop_cfg['EXT_INPUT_FIELDNAME']}"
    )
    chop_fields_map[chop_cfg["EXT_INPUT_FIELDNAME"]] = "ext_input"

interface = LFADSInterface(
    window=WIN_LEN,
    overlap=OLAP_LEN,
    max_offset=MAX_OFF,
    chop_margins=CHOP_MARG,
    random_seed=RAND_SEED,
    chop_fields_map=chop_fields_map,
)

  
ds_name = (
    "lfads_" + NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + ".h5"
)
ds_obj_name = (
    "lfads_" + NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + "_fulldataset.pkl"
)

INTERFACE_FILE = os.path.join(
    pkl_dir,
    NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + "_interface.pkl",
)

# save deemg input and dataset for each session

DATA_FILE = os.path.join(lfads_save_dir, ds_name)
DATASET_OBJECT = os.path.join(lfads_save_dir, ds_obj_name)

# --- chop and save h5 dataset
interface.chop_and_save(chop_df, DATA_FILE, overwrite=True)

# --- save original dataframe
with open(DATASET_OBJECT,'wb') as outf:
    logger.info(f"Original data file {DATASET_OBJECT} saved to pickle.")
    pickle.dump(dataset, outf)



# --- save interface object
with open(INTERFACE_FILE, "wb") as rfile:
    logger.info(f"Interface {INTERFACE_FILE} saved to pickle.")
    pickle.dump(interface, rfile)

# %%
