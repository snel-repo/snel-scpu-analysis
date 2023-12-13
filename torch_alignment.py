# Script to change lfads-tf2 alignment matrices to lfads torch
# Concatenate the data file and alignment matrices file and rename fields to:

# DATA FILE
# "train_data" -> "train_encod_data"
# "valid_data" -> "valid_encod_data"
# "train_data" -> "train_recon_data" (duplicate)
# "valid_data" -> "valid_recon_data" (duplicate)

# ALIGNMENT MATRICES FILE
# concatenate with lfads_gran_{sess}_ALL_spikes_10.h5/bias (field name "readout_bias")
# concatenate with lfads_gran_{sess}_ALL_spikes_10.h5/matrix (field name "readin_weight")


# Run this script in /snel/share/share/derived/auyong/nwb_lfads/runs/binsize_10ms_pcr_high_reg_ALL_initials/
# To run this script, type: python torch_alignment.py <your_initials>

import h5py
import sys
import os
import numpy as np


cat_name = "gran"
sess_numbers = ["013", "023", "028", "030", "031", "033"]
initials = str(sys.argv[1])
data_dir = f"/snel/share/share/derived/auyong/nwb_lfads/runs/binsize_10ms_pcr_high_reg_ALL_{initials}"

os.chdir(data_dir)

def concat_for_torch(cat_name, sess_numbers, run_type):
    align_file = f"{cat_name}/alignment_matrices/{run_type}/pcr_alignment_torch.h5"
    
    
    for sess in sess_numbers:
        if run_type == "spikes":
            dataset_file = f"{cat_name}/datasets/lfads_gran_{sess}_ALL_{run_type}_10.h5"
        elif run_type == "emg":
            dataset_file = f"{cat_name}/datasets/lfads_gran_{sess}_{run_type}_10.h5"
        with h5py.File(dataset_file, "r") as dataset:

            train_encod_data = dataset["train_data"][:]
            valid_encod_data = dataset["valid_data"][:]
            train_recon_data = dataset["train_data"][:]
            valid_recon_data = dataset["valid_data"][:]
        try:
            readout_bias, readin_weight = retrieve_weight_bias(align_file, sess, run_type)
        except ValueError as e:
            print(e)
            break

        kwargs = dict(dtype='float32', compression='gzip')
        with h5py.File(f"{data_dir}/lfads_torch_readin{sess}_{run_type}.h5", 'w') as h5f:

            h5f.create_dataset('train_encod_data', data=train_encod_data, **kwargs)
            h5f.create_dataset('valid_encod_data', data=valid_encod_data, **kwargs)
            h5f.create_dataset('train_recon_data', data=train_recon_data, **kwargs)
            h5f.create_dataset('valid_recon_data', data=valid_recon_data, **kwargs)
            h5f.create_dataset('readout_bias', data=readout_bias, **kwargs)
            h5f.create_dataset('readin_weight', data=readin_weight, **kwargs)
            print(f"File saved: {data_dir}/lfads_torch_readin{sess}_{run_type}.h5")
        


def retrieve_weight_bias(align_file, sess, run_type):
    if run_type == "spikes":
        key_name = f"lfads_gran_{sess}_ALL_{run_type}_10.h5"
    elif run_type == "emg":
        key_name = f"lfads_gran_{sess}_{run_type}_10.h5"

    readout_bias, readin_weight = None, None
    with h5py.File(align_file, "r") as align:
        if key_name in align:
            # Retrieve the 'matrix' dataset
            if 'matrix' in align[key_name]:
               readin_weight = align[key_name]['matrix'][:]

            # Retrieve the 'bias' dataset
            if 'bias' in align[key_name]:
                readout_bias = align[key_name]['bias'][:]
        else:
            print(f"Error: {key_name} not in {align_file}")
    if readout_bias is None or readin_weight is None:
        raise ValueError("Error: bias or weight is None")
    return readout_bias, readin_weight

concat_for_torch(cat_name, sess_numbers, "spikes")
concat_for_torch(cat_name, sess_numbers, "emg")
