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

# --- handle system inputs
expt_name = "NP_AAV6-2_ReaChR_184500"

# === begin SCRIPT PARAMETERS ==========================

lfads_dataset_cfg = [
    {
        "DATASET": {
            "NAME": expt_name,
            "CONDITION_SEP_FIELD": None,  # continuous
            "ALIGN_LIMS": None,
            "ARRAY_SELECT": "ALL",  # 'R', 'L', 'both'
            "BIN_SIZE": 2, # 4,
            "SPK_KEEP_THRESHOLD": None,  # 15,
            "SPK_XCORR_THRESHOLD": 0.1,
            "EXCLUDE_TRIALS": [],
            "EXCLUDE_CONDITIONS": [],
            "EXCLUDE_CHANNELS": [],
        }
    },
    {
        "CHOP_PARAMETERS": {
            #"TYPE": "emg",
            #"DATA_FIELDNAME": "model_emg",
            "TYPE": "spikes",
            "DATA_FIELDNAME": "spikes",
            "USE_EXT_INPUT": False,
            "EXT_INPUT_FIELDNAME": "",
            "WINDOW": 200,  # ms
            "OVERLAP": 50,  # ms
            "MAX_OFFSET": 0,
            "RANDOM_SEED": 0,
            "CHOP_MARGINS": 0,
        }
    },
]


# -- paths
base_name = "test_binsize_2ms"
ds_base_dir = "/snel/share/share/derived/scpu_snel/NWB/"
lfads_save_dir = f"/snel/share/share/derived/scpu_snel/nwb_lfads/runs/{base_name}/{expt_name}/datasets/"

# === end SCRIPT PARAMETERS ==========================


ds_path = os.path.join(ds_base_dir, expt_name + ".nwb")
# --- load dataset from NWB
logger.info(f"Loading {expt_name} from NWB")
dataset = NWBDataset(ds_path, split_heldout=False)

# --- preprocess spiking data


ld_cfg = lfads_dataset_cfg[0]["DATASET"]

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

if len(drop_spk_names) > 0:
    chop_df.drop(columns=drop_spk_names, axis=1, level=1, inplace=True)

# --- create save dirs if they do not exist
pkl_dir = os.path.join(lfads_save_dir, "pkls")
if os.path.exists(lfads_save_dir) is not True:
    os.makedirs(lfads_save_dir)
if os.path.exists(pkl_dir) is not True:
    os.makedirs(pkl_dir)

# --- initialize chop interface

chop_cfg = lfads_dataset_cfg[1]["CHOP_PARAMETERS"]

# -- extract relevant config params
WIN_LEN = chop_cfg["WINDOW"]
OLAP_LEN = chop_cfg["OVERLAP"]
MAX_OFF = chop_cfg["MAX_OFFSET"]
CHOP_MARG = chop_cfg["CHOP_MARGINS"]
RAND_SEED = chop_cfg["RANDOM_SEED"]
TYPE = chop_cfg["TYPE"]
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
if TYPE == "emg":
    chan_keep_mask = emg_keep_mask.tolist()
    ds_name = "lfads_" + NAME + "_" + TYPE + "_" + str(BIN_SIZE) + ".h5"
    ds_obj_name = "lfads_" + NAME + "_" + TYPE + "_" + str(BIN_SIZE) + "_fulldataset.pkl"
    yaml_name = "cfg_" + NAME + "_" + TYPE + "_" + str(BIN_SIZE) + ".yaml"
    INTERFACE_FILE = os.path.join(
        pkl_dir,
        NAME + "_" + TYPE + "_" + str(BIN_SIZE) + "_interface.pkl",
    )
elif chop_cfg["TYPE"] == "spikes":    
    ds_name = (
        "lfads_" + NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + ".h5"
    )
    ds_obj_name = (
        "lfads_" + NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + "_fulldataset.pkl"
    )
    yaml_name = (
        "cfg_" + NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + ".yaml"
    )
    INTERFACE_FILE = os.path.join(
        pkl_dir,
        NAME + "_" + ARRAY_SELECT + "_" + TYPE + "_" + str(BIN_SIZE) + "_interface.pkl",
    )

# save deemg input and dataset for each session

DATA_FILE = os.path.join(lfads_save_dir, ds_name)
YAML_FILE = os.path.join(lfads_save_dir, yaml_name)
DATASET_OBJECT = os.path.join(lfads_save_dir, ds_obj_name)

# --- chop and save h5 dataset
interface.chop_and_save(chop_df, DATA_FILE, overwrite=True)

# --- save original dataframe
with open(DATASET_OBJECT,'wb') as outf:
    logger.info(f"Original data file {DATASET_OBJECT} saved to pickle.")
    pickle.dump(dataset, outf)

# --- save yaml config file
with open(YAML_FILE, "w") as yamlfile:
    logger.info(f"YAML {YAML_FILE} saved to pickle.")
    data1 = yaml.dump(lfads_dataset_cfg, yamlfile)
    yamlfile.close()

# --- save interface object
with open(INTERFACE_FILE, "wb") as rfile:
    logger.info(f"Interface {INTERFACE_FILE} saved to pickle.")
    pickle.dump(interface, rfile)

# %%
