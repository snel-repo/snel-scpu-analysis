import h5py
import typing 
import numpy as np
import pandas as pd
import yaml 
from snel_toolkit.datasets.nwb import NWBDataset

# load YAML file
yaml_config_path = "../configs/lfads_dataset_cfg.yaml"
lfads_dataset_cfg = yaml.load(open(yaml_config_path), Loader=yaml.FullLoader)

path_config = lfads_dataset_cfg["PATH_CONFIG"]
ld_cfg = lfads_dataset_cfg["DATASET"]
merge_config = lfads_dataset_cfg["MERGE_PARAMETERS"]
BIN_SIZE = ld_cfg["BIN_SIZE"]

def get_train_valid_inds(original_h5: str, torch_outputs: h5py._hl.files.File, lfads_torch_outputs_path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    original_h5_data = h5py.File(original_h5)
    train_inds = original_h5_data['train_inds'][()]
    valid_inds = original_h5_data['valid_inds'][()]
    # check if torch output already has train/valid inds
    if 'train_inds' not in torch_outputs.keys():
        with h5py.File(lfads_torch_outputs_path,'a') as torch_output_data:
            torch_output_data.create_dataset('train_inds',data=train_inds)
            torch_output_data.create_dataset('valid_inds',data=valid_inds)

    return train_inds, valid_inds


def combine_train_valid_outputs(torch_outputs: h5py._hl.files.File,
                                train_inds: np.ndarray, 
                                valid_inds: np.ndarray,
                                merge_config: typing.Dict[str, str]) ->\
                                typing.Dict[str, np.ndarray]:

    n_batch = train_inds.size + valid_inds.size
    data_dict = {} # dict with combined data
    for torch_name, snel_toolkit_name in merge_config.items():
        # key is torch names, val is what snel_toolkit name should be
        train_output = torch_outputs[f'train_{torch_name}'][()]
        valid_output = torch_outputs[f'valid_{torch_name}'][()]
        full_output = np.empty((n_batch, train_output.shape[1], train_output.shape[2]))
        full_output[train_inds,:,:] = train_output
        full_output[valid_inds,:,:] = valid_output
        data_dict[torch_name] = full_output
    
    return data_dict

def merge_with_original_df(merged_df: pd.DataFrame, dataset: NWBDataset):
    for key in merged_df.columns.levels[0].to_list():
        if key == "lfads_rates":
            chan_names = dataset.data['spikes'].columns.values
        else: 
            chan_names = np.arange(merged_df[key].shape[1])
        if key in dataset.data.keys():
            dataset.data[key] = merged_df[key]
        else:
            dataset.add_continuous_data(
                merged_df[key].values,
                key,
                chan_names=chan_names,
            )

def get_event_start_stop_ix(win_len_ms: int, pre_buffer_ms: int, event_id: int, dataset: NWBDataset) -> typing.Tuple[int, int]:
    win_len = win_len_ms / BIN_SIZE
    event_start_time = dataset.trial_info.iloc[event_id].start_time - pd.to_timedelta(pre_buffer_ms, unit="ms")
    start_ix = dataset.data.index.get_loc(event_start_time, method='nearest')
    stop_ix = int(start_ix + win_len)
    
    return start_ix, stop_ix