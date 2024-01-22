import h5py
import typing 
def get_train_valid_inds(original_h5, torch_outputs, lfads_torch_outputs_path):
    original_h5_data = h5py.File(original_h5)
    train_inds = original_h5_data['train_inds'][()]
    valid_inds = original_h5_data['valid_inds'][()]
    # check if torch output already has train/valid inds
    if 'train_inds' not in torch_outputs.keys():
        with h5py.File(lfads_torch_outputs_path,'a') as torch_output_data:
            torch_output_data.create_dataset('train_inds',data=train_inds)
            torch_output_data.create_dataset('valid_inds',data=valid_inds)

    return train_inds, valid_inds


def combine_train_valid_outputs(torch_outputs, train_inds, valid_inds, merge_config):

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
