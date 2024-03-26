# from nlb_tools.nwb_interface import NWBDataset
from snel_toolkit.datasets.nwb import NWBDataset

# from rds.structures import DataWrangler
from snel_toolkit.datasets.base import DataWrangler
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
# from rds.decoding import prepare_decoding_data
import scipy.signal as signal
from os import path
from tqdm import tqdm
import _pickle as pickle
import logging
import h5py
import glob
import os
import sys
import yaml

# --- setup logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- handle system inputs
ds_base_name = sys.argv[1]
session_ids = sys.argv[2]
session_ids = session_ids.split(",")
session_ids[0] = session_ids[0].replace("[", "")
session_ids[-1] = session_ids[-1].replace("]", "")

# === begin SCRIPT PARAMETERS ==========================

# -- script control options
debug = False
save_align_mats = True
torch_output = True #whether to save alignment matrices in torch format (i.e. readin weight and readout bias)

# -- analysis params
BIN_SIZE = 10  # ms
GAUSS_WIDTH_MS = 30  # width of gaussian kernel(ms)
CLIP_Q_CUTOFF = 0.99  # emg quantile to set clipping
SCALE_Q_CUTOFF = 0.95  # emg quantile to set unit scaling
XCORR_THRESHOLD = 0.1  # spk/s
ARRAY_SELECT = "ALL"  # which array data to model, if "ALL", use both arrays

NUM_SPK_PCS = 20
NUM_EMG_PCS = 10
L2_SCALE = 1e-2 #1e-0

model_emg_field = "model_emg"

# -- paths
align_file_suffix = "" # "_low_reg"
base_name = f"binsize_10ms_pcr_high_reg_{ARRAY_SELECT}_cw"
#base_name = f"binsize_4ms_{ARRAY_SELECT}"
nwb_cache_dir = f"/snel/share/share/tmp/scratch/cbwash2/auyong_nwb/{base_name}/"
ds_base_dir = "/snel/share/share/derived/auyong/NWB/"

# --- create save directories for PCR alignment matrices
base_pcr_save_dir = (
    f"/snel/share/share/derived/auyong/nwb_lfads/runs/{base_name}/{sys.argv[1]}/"
)
base_align_mat_dir = os.path.join(base_pcr_save_dir, "alignment_matrices")
emg_align_mat_dir = os.path.join(base_align_mat_dir, "emg")
spk_align_mat_dir = os.path.join(base_align_mat_dir, "spikes")


# === end SCRIPT PARAMETERS ==========================

# === begin FUNCTION DEFINITIONS ==========================
def _return_field_names(dataset, field):
    """returns field_names"""
    field_names = dataset.data[field].columns.values.tolist()
    return field_names


def aligned_cycle_averaging(dataset, dw, field, field_names=None):
    """perform cycle aligned averaging"""
    if field_names is None:
        field_names = _return_field_names(ds, field)
    cycle_avg = np.full((dw._t_df.align_time.unique().size, len(field_names)), np.nan)
    cycle_data = np.full(
        (
            dw._t_df.align_time.unique().size,
            len(field_names),
            dw._t_df.trial_id.nunique(),
        ),
        np.nan,
    )

    for i, fname in enumerate(field_names):
        cycle_aligned_df = dw.pivot_trial_df(dw._t_df, values=(field, fname))
        cycle_avg[:, i] = cycle_aligned_df.mean(axis=1, skipna=True)
        cycle_data[:, i, :] = cycle_aligned_df
    return cycle_avg, cycle_data


def concat_sessions(all_avg, all_means):
    # --- concatenate all sessions together to create "global" space
    global_avg = np.concatenate(all_avg, axis=1)
    global_means = np.concatenate(all_means, axis=1)

    return global_avg, global_means


def fit_global_pcs(global_avg, global_means, num_pcs, fit_ix):
    from sklearn.decomposition import PCA

    pca_obj = PCA(n_components=num_pcs)

    # --- mean center
    mean_cent_global_avg = global_avg[fit_ix, :] - global_means

    # --- fit pca
    global_pcs = pca_obj.fit_transform(mean_cent_global_avg)

    return pca_obj, global_pcs


def fit_session_readins(all_avg, all_means, global_pcs, fit_ix, l2_scale=0):
    all_W = []
    all_b = []
    all_b_out = []
    for sess_avg, sess_means in zip(all_avg, all_means):
        lr = Ridge(alpha=l2_scale, fit_intercept=False)
        lr.fit(sess_avg[fit_ix, :] - sess_means, global_pcs)

        # -- weights [ chans x pcs ]
        W = lr.coef_.T

        # -- bias [ pcs ]
        b = sess_means
        all_b_out.append(b)
        # readin layer adds bias after matmul
        b = -1 * np.dot(b, W)
        all_W.append(W)
        all_b.append(b)

    return all_W, all_b, all_b_out


# === end FUNCTION DEFINITIONS ==========================


data_wranglers = []
all_spk_cycle_avg = []
all_spk_chan_means = []
all_emg_cycle_avg = []
all_emg_chan_means = []
all_ds = []
for session_id in session_ids:
    ds_name = f"{ds_base_name}_{session_id}"
    ds_path = path.join(ds_base_dir, ds_name + ".nwb")
    # --- load dataset from NWB
    logger.info(f"Loading {ds_name} from NWB")
    dataset = NWBDataset(ds_path, split_heldout=False)

    # --- spiking data pre-processing (i.e., array select, xcorr rejection, smoothing)
    #import pdb; pdb.set_trace()
    # -- generate mask to select which neurons to include in analysis
    spk_keep_mask = np.array(
        [ARRAY_SELECT in val for val in dataset.unit_info.group_name.values]
    )
    if ARRAY_SELECT == "ALL":
        spk_keep_mask[:] = True

    # -- xcorr rejection
    pair_xcorr, chan_names_to_drop = dataset.get_pair_xcorr(
        "spikes", threshold=XCORR_THRESHOLD, zero_chans=True
    )

    # -- update keep mask
    spk_keep_mask[chan_names_to_drop] = False

    spk_names = dataset.data.spikes.columns.values
    drop_spk_names = spk_names[~spk_keep_mask]
    keep_spk_names = spk_names[spk_keep_mask]
    # -- drop channels
    logger.info(f"Keep spike channels: {np.sum(spk_keep_mask)}/{spk_keep_mask.size}")
    if type(np.any(drop_spk_names)) == int:
        dataset.data.drop(
            columns=drop_spk_names.tolist(), axis=1, level=1, inplace=True
        )

    # -- smooth spikes
    dataset.smooth_spk(
        GAUSS_WIDTH_MS, signal_type="spikes", name=f"smooth_{GAUSS_WIDTH_MS}ms"
    )
    smooth_spk_field = f"spikes_smooth_{GAUSS_WIDTH_MS}ms"
    # --- emg data pre-processing (i.e., clipping, scaling, smoothing)
    clip_emg = dataset.data.emg.copy(deep=True)
    emg_names = clip_emg.columns.values
    clip_q = [CLIP_Q_CUTOFF] * emg_names.size
    for clip_q_cutoff, emg_name in zip(clip_q, emg_names):
        # -- clipping
        chan_emg = clip_emg[emg_name]
        clip_q = chan_emg.quantile(clip_q_cutoff)
        clip_chan_emg = chan_emg.clip(upper=clip_q)
        # -- scaling
        scale_q = clip_chan_emg.quantile(SCALE_Q_CUTOFF)
        scale_emg = clip_chan_emg / scale_q
        dataset.data[(model_emg_field, emg_name)] = scale_emg

    # -- smooth emg
    dataset.smooth_spk(
        GAUSS_WIDTH_MS, signal_type=model_emg_field, name=f"smooth_{GAUSS_WIDTH_MS}ms"
    )
    smooth_emg_field = f"{model_emg_field}_smooth_{GAUSS_WIDTH_MS}ms"

    # --- resample dataset from 1ms bins to desired bin size used for analysis
    dataset.resample(BIN_SIZE)
    all_ds.append(dataset)
    fig = plt.figure(figsize=(14,5)); 
    
    savedir = f"/snel/share/share/tmp/cbwash2/scratch/spinal_analyses/{ds_base_name}/"
    plt_savedir = os.path.join(savedir, "overview") 
    if not os.path.isdir(plt_savedir):
        logger.info(f"creating {plt_savedir}")
        os.makedirs(plt_savedir)
    fig, axs = plt.subplots(4,1, figsize=(14,10)); 
    axs[0].pcolor(dataset.data.spikes_smooth_30ms.values.T); 
    axs[0].set_title(ds_name); 
    r_units = dataset.unit_info.location.str.contains("R").sum(); 
    l_units = dataset.unit_info.shape[0] - r_units; 
    axs[0].set_ylabel(f"R: {r_units}, L: {l_units}"); 
    
    axs[1].plot(dataset.data[smooth_emg_field]["RSL"].values)
    axs[1].plot(dataset.data[smooth_emg_field]["LSL"].values)
    axs[2].plot(dataset.data[smooth_emg_field]["RBA"].values)
    axs[2].plot(dataset.data[smooth_emg_field]["RBP"].values)
    axs[3].plot(dataset.data[smooth_emg_field]["LBA"].values)
    axs[3].plot(dataset.data[smooth_emg_field]["LBP"].values)
    plt.savefig(os.path.join(plt_savedir, f"{ds_name}_overview.png"), dpi=150)
    #import pdb; pdb.set_trace()
    # --- save datasets
    # -- create cache dir if is does not exist
    if not os.path.isdir(nwb_cache_dir):
        logger.info(f"Creating {nwb_cache_dir}")
        os.makedirs(nwb_cache_dir)
    # -- dump dataset object to pickle file
    ds_savepath = path.join(nwb_cache_dir, "nlb_" + ds_name + ".pkl")
    with open(ds_savepath, "wb") as rfile:
        logger.info(f"Dataset {ds_name} saved to pickle.")
        pickle.dump(dataset, rfile, protocol=4)

    group_field = "condition_id"
    emg_field = smooth_emg_field
    spk_field = smooth_spk_field

    emg_names = dataset.data.emg.columns.values
    # side_emg_names = emg_names

    # side_spk_names = keep_spk_names
    spk_names = keep_spk_names

    # --- choose ignored trials
    # get trial info from dataset
    ti = dataset.trial_info
    excluded_trials = [
        0,
        1,
        2,
        3,
        ti.trial_id.iloc[-2],
        ti.trial_id.iloc[-1],
    ]  # first 4 and last 2 cycles
    excluded = ti.trial_id == -1  # creating an boolean index
    for ex_t in excluded_trials:
        excluded[ti.trial_id == ex_t] = True  # mark to exlude if in list
    ignore_trials = excluded

    # --- align data
    dw = DataWrangler(dataset)
    dw.make_trial_data(
        name="onset",
        align_field="start_time",
        align_range=(-100, 600),  # (-100, 600),  # (0, 800)  # (-200, 950),
        ignored_trials=ignore_trials,
        allow_overlap=True,
        set_t_df=True,
    )
    # --- cycle average of spiking data
    spk_cycle_avg, _ = aligned_cycle_averaging(
        dataset, dw, spk_field, field_names=spk_names
    )
    spk_chan_means = np.nanmean(spk_cycle_avg, axis=0)[np.newaxis, :]
    all_spk_chan_means.append(spk_chan_means)
    all_spk_cycle_avg.append(spk_cycle_avg)

    # --- cycle average of emg data
    emg_cycle_avg, _ = aligned_cycle_averaging(
        dataset, dw, emg_field, field_names=emg_names
    )
    emg_chan_means = np.nanmean(emg_cycle_avg, axis=0)[np.newaxis, :]
    all_emg_chan_means.append(emg_chan_means)
    all_emg_cycle_avg.append(emg_cycle_avg)

#import pdb; pdb.set_trace()
# --- create global spk and emg space
global_spk_cycle_avg, global_spk_chan_means = concat_sessions(
    all_spk_cycle_avg, all_spk_chan_means
)
global_emg_cycle_avg, global_emg_chan_means = concat_sessions(
    all_emg_cycle_avg, all_emg_chan_means
)


# --- fit pca on global pcs
spk_fit_ix = ~np.any(np.isnan(global_spk_cycle_avg), axis=1)
pca_spk, global_spk_pcs = fit_global_pcs(
    global_spk_cycle_avg, global_spk_chan_means, NUM_SPK_PCS, spk_fit_ix
)

emg_fit_ix = ~np.any(np.isnan(global_emg_cycle_avg), axis=1)
pca_emg, global_emg_pcs = fit_global_pcs(
    global_emg_cycle_avg, global_emg_chan_means, NUM_EMG_PCS, spk_fit_ix
)


# --- create h5 files to save PCR readin matrices if desired
if save_align_mats:
    # --- create directories to store alignment matrices if they do not exist
    if not os.path.isdir(emg_align_mat_dir):
        logger.info(f"Creating {emg_align_mat_dir}")
        os.makedirs(emg_align_mat_dir)
    if not os.path.isdir(spk_align_mat_dir):
        logger.info(f"Creating {spk_align_mat_dir}")
        os.makedirs(spk_align_mat_dir)

    align_filename = f"pcr_alignment{align_file_suffix}.h5"
    if torch_output:
        align_filename_torch = f"pcr_alignment{align_file_suffix}_torch.h5"
    lfads_dataset_prefix = "lfads"
    hf_spk = h5py.File(os.path.join(spk_align_mat_dir, align_filename), "w")
    hf_emg = h5py.File(os.path.join(emg_align_mat_dir, align_filename), "w")
    if torch_output:
        hf_spk_torch = h5py.File(os.path.join(spk_align_mat_dir, align_filename_torch), "w")
        hf_emg_torch = h5py.File(os.path.join(emg_align_mat_dir, align_filename_torch), "w")

all_spk_W, all_spk_b, all_spk_b_out = fit_session_readins(
    all_spk_cycle_avg, all_spk_chan_means, global_spk_pcs, spk_fit_ix, L2_SCALE
)
all_emg_W, all_emg_b, all_emg_b_out = fit_session_readins(
    all_emg_cycle_avg, all_emg_chan_means, global_emg_pcs, emg_fit_ix, L2_SCALE
)

for i, (sess_id, W_spk, b_spk, b_spk_out, W_emg, b_emg, b_emg_out) in enumerate(
    zip(
        session_ids,
        all_spk_W,
        all_spk_b,
        all_spk_b_out,
        all_emg_W,
        all_emg_b,
        all_emg_b_out
    )
):
    ds_name = f"{ds_base_name}_{sess_id}"
    if debug:
        logger.info(f"==== {ds_name} PCR Readins ====")
        logger.info(f"Spk weight matrix dim: {W_spk.shape}")
        logger.info(f"Spk bias matrix dim: {np.squeeze(b_spk).shape}")

        logger.info(f"EMG weight matrix dim: {W_emg.shape}")
        logger.info(f"EMG bias matrix dim: {np.squeeze(b_emg).shape}")

    if save_align_mats:
        emg_group_name = (
            "_".join((lfads_dataset_prefix, ds_name, "emg", str(BIN_SIZE))) + ".h5"
        )
        spk_group_name = (
            "_".join(
                (lfads_dataset_prefix, ds_name, ARRAY_SELECT, "spikes", str(BIN_SIZE))
            )
            + ".h5"
        )
        if debug:
            logger.info(f"Creating spk  alignment matrices for {spk_group_name}")
            logger.info(f"Creating emg alignment matrices for {emg_group_name}")

        emg_group = hf_emg.create_group(emg_group_name)
        emg_group.create_dataset("matrix", data=W_emg)
        emg_group.create_dataset("bias", data=np.squeeze(b_emg))

        spk_group = hf_spk.create_group(spk_group_name)
        spk_group.create_dataset("matrix", data=W_spk)
        spk_group.create_dataset("bias", data=np.squeeze(b_spk))

        if torch_output:
            emg_group = hf_emg_torch.create_group(emg_group_name)
            emg_group.create_dataset("matrix", data=W_emg)
            emg_group.create_dataset("bias", data=np.squeeze(b_emg_out))

            spk_group = hf_spk_torch.create_group(spk_group_name)
            spk_group.create_dataset("matrix", data=W_spk)
            spk_group.create_dataset("bias", data=np.squeeze(b_spk_out))

if save_align_mats:
    hf_spk.close()
    hf_emg.close()
    if torch_output:
        hf_spk_torch.close()
        hf_emg_torch.close()

if debug:
    fig = plt.figure()

    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    def plot_pcs(ax, pcs, **kwargs):
        ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], **kwargs)

    cm = colormap.Dark2

    plot_pcs(ax, global_spk_pcs, label="Global", color="k", linewidth=2)
    plot_pcs(ax2, global_emg_pcs, label="Global", color="k", linewidth=2)
    for i, (
        session_id,
        sess_spk_cycle_avg,
        sess_emg_cycle_avg,
        W_spk,
        b_spk,
        W_emg,
        b_emg,
    ) in enumerate(
        zip(
            session_ids,
            all_spk_cycle_avg,
            all_emg_cycle_avg,
            all_spk_W,
            all_spk_b,
            all_emg_W,
            all_emg_b,
        )
    ):

        ds_name = f"{ds_base_name}_{session_id}"
        sess_spk_cycle_pcs = np.matmul(sess_spk_cycle_avg[spk_fit_ix, :], W_spk) + b_spk
        sess_emg_cycle_pcs = np.matmul(sess_emg_cycle_avg[emg_fit_ix, :], W_emg) + b_emg
        plot_pcs(
            ax,
            sess_spk_cycle_pcs,
            label=ds_name,
            color=cm(float(i) / len(session_ids)),
            alpha=0.4,
        )
        plot_pcs(
            ax2,
            sess_emg_cycle_pcs,
            label=ds_name,
            color=cm(float(i) / len(session_ids)),
            alpha=0.4,
        )

    plt.legend()
    ax.set_xlabel("PC1")
    ax2.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax2.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax2.set_zlabel("PC3")
    ax.set_title(f"{ARRAY_SELECT} Spikes")
    ax2.set_title("EMG")
    plt.suptitle("Multi-session PCR")

    plt.figure()
    plt.plot(
        np.cumsum(pca_spk.explained_variance_ratio_), "-o", color="g", label="spikes"
    )
    plt.plot(np.cumsum(pca_emg.explained_variance_ratio_), "-o", color="b", label="emg")
    plt.xlabel("Number of PCs")
    plt.ylabel("Total Variance Explained")
    plt.legend()
    plt.show()
# import pdb

# pdb.set_trace()
