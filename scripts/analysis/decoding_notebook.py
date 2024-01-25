"""
PURPOSE: Decode the latent factors using a linear decoder

REQUIREMENTS: merged pkl file from merge_chopped_torch_outputs.py
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
# Load YAML

# load YAML file
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
path_config, ld_cfg, merge_config = load_cfgs(yaml_config_path)

