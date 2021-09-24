import os
import sys
import torch
import logging
import pickle
import csv
import glob
import shutil
import numpy as np
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.processing.PLDA_LDA import StatObject_SB
from speechbrain.processing import diarization as diar
from speechbrain.utils.DER import DER
# import sklearn

np.random.seed(1234)

# Logger setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

def compute_embeddings(wavs, lens):
    """Definition of the steps for computation of embeddings from the waveforms
    """
    with torch.no_grad():
        wavs = wavs.to(params["device"])
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, lens)
        emb = params["embedding_model"](feats, lens)
        emb = params["mean_var_norm_emb"](
            emb, torch.ones(emb.shape[0], device=params["device"])
        )

    return emb

def embedding_computation_loop(split, set_loader, stat_file):
    """Extracts embeddings for a given dataset loader
    """

    # Extract embeddings (skip if already done)
    if not os.path.isfile(stat_file):
        logger.debug("Extracting deep embeddings and diarizing")
        embeddings = np.empty(shape=[0, params["emb_dim"]], dtype=np.float64)
        modelset = []
        segset = []

        # Different data may have different statistics
        params["mean_var_norm_emb"].count = 0

        for batch in set_loader:  # t:
            ids = batch.id
            wavs, lens = batch.sig

            mod = [x for x in ids]
            seg = [x for x in ids]
            modelset = modelset + mod
            segset = segset + seg

            # Embedding computation
            emb = (
                compute_embeddings(wavs, lens)
                .contiguous()
                .squeeze(1)
                .cpu()
                .numpy()
            )
            embeddings = np.concatenate((embeddings, emb), axis=0)

        modelset = np.array(modelset, dtype="|O")
        segset = np.array(segset, dtype="|O")

        # Intialize variables for start, stop and stat0
        s = np.array([None] * embeddings.shape[0])
        b = np.array([[1.0]] * embeddings.shape[0])

        stat_obj = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=embeddings,
        )
        logger.debug("Saving Embeddings...")
        stat_obj.save_stat_object(stat_file)

    else:
        logger.debug("Skipping embedding extraction (as already present)")
        logger.debug("Loading previously saved embeddings")

        with open(stat_file, "rb") as in_file:
            stat_obj = pickle.load(in_file)

    return stat_obj


def dataio_prep(hparams, csv_file):
    "Creates the datasets and their data processing pipelines."

    # 1. Datasets
    data_folder = hparams["data_folder"]
    dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_file, replacements={"data_root": data_folder},
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item([dataset], audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys([dataset], ["id", "sig"])

    # 4. create dataloader:
    dataloader = sb.dataio.dataloader.make_dataloader(
        dataset, **params["dataloader_opts"]
    )

    return dataloader
    

def diarize_dataset(full_csv, split_type, n_lambdas, pval, n_neighbors=10):
    """Diarizes all the recordings in a given dataset
    """

    # Prepare `spkr_info` only once when Oracle num of speakers is selected
    if params["oracle_n_spkrs"] is True:
        full_ref_rttm_file = (
            params["ref_rttm_dir"] + "/fullref_synd_" + split_type + ".rttm"
        )

        rttm = diar.read_rttm(full_ref_rttm_file)

        spkr_info = list(  # noqa F841
            filter(lambda x: x.startswith("SPKR-INFO"), rttm)
        )

    # Get all recording IDs in this dataset
    A = [row[0].rstrip().split("_")[0] for row in full_csv]
    all_rec_ids = list(set(A[1:]))
    all_rec_ids.sort()

    N = str(len(all_rec_ids))
    split = "Synd_" + split_type
    i = 1

    # Setting eval modality
    params["embedding_model"].eval()
    msg = "Diarizing " + split_type + " set"
    logger.info(msg)
    for rec_id in tqdm(all_rec_ids):

        tag = "[" + str(split_type) + ": " + str(i) + "/" + N + "]"
        i = i + 1

        msg = "Diarizing %s : %s " % (tag, rec_id)
        logger.debug(msg)

        if not os.path.exists(os.path.join(params["embedding_dir"], split)):
            os.makedirs(os.path.join(params["embedding_dir"], split))

        diary_stat_file = os.path.join(
            params["embedding_dir"], split, rec_id + "_xv_stat.pkl"
        )

        # Prepare a csv for a recording
        new_csv_file = os.path.join(
            params["embedding_dir"], split, rec_id + ".csv"
        )
        diar.prepare_subset_csv(full_csv, rec_id, new_csv_file)

        # Setup a dataloader for above one recording (above csv)
        diary_set_loader = dataio_prep(params, new_csv_file)

        # Putting modules on the device
        params["compute_features"].to(params["device"])
        params["mean_var_norm"].to(params["device"])
        params["embedding_model"].to(params["device"])
        params["mean_var_norm_emb"].to(params["device"])

        # Compute Embeddings
        diary_obj = embedding_computation_loop(
            "diary", diary_set_loader, diary_stat_file
        )

        # Perform spectral clustering
        out_rttm_dir = os.path.join(params["sys_rttm_dir"], split)
        if not os.path.exists(out_rttm_dir):
            os.makedirs(out_rttm_dir)
        out_rttm_file = out_rttm_dir + "/" + rec_id + ".rttm"

        if params["oracle_n_spkrs"] is True:
            # Oracle num of speakers
            num_spkrs = diar.get_oracle_num_spkrs(rec_id, spkr_info)
        else:
            if params["affinity"] == "nn":
                # Num of speakers tunned on dev set
                num_spkrs = n_lambdas
            else:
                # Will be estimated using max eigen gap for cos based affinity
                num_spkrs = None

        diar.do_spec_clustering(
            diary_obj,
            out_rttm_file,
            rec_id,
            num_spkrs,
            pval,
            params["affinity"],
            n_neighbors,
        )

    # Concatenate individual RTTM files
    # This is not needed but just staying with the standards
    concate_rttm_file = out_rttm_dir + "/sys_output.rttm"

    logger.debug("Concatenating individual RTTM files...")
    with open(concate_rttm_file, "w") as cat_file:
        for f in glob.glob(out_rttm_dir + "/*.rttm"):
            if f == concate_rttm_file:
                continue
            with open(f, "r") as indi_rttm_file:
                shutil.copyfileobj(indi_rttm_file, cat_file)

    msg = "The system generated RTTM file for %s set : %s" % (
        split_type,
        concate_rttm_file,
    )
    logger.debug(msg)

    return concate_rttm_file


def dev_p_tuner(full_csv, split_type):
    """Tuning p_value affinity matrix
    """

    DER_list = []
    prange = np.arange(0.002, 0.015, 0.001)

    n_lambdas = None
    for p_v in prange:
        # Process whole dataset for value of p_v
        concate_rttm_file = diarize_dataset(
            full_csv, split_type, n_lambdas, p_v
        )

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_synd_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        DER_list.append(DER_)

    # Take p_val that gave minimum DER on Dev dataset
    tuned_p_val = prange[DER_list.index(min(DER_list))]

    return tuned_p_val


def dev_nn_tuner(full_csv, split_type):
    """Tuning n_neighbors on dev set.
    Assuming oracle num of speakers.
    """

    DER_list = []
    pval = None
    for nn in range(3, 15):

        # Fix this later. Now assuming oracle num of speakers
        n_lambdas = 2

        # Process whole dataset for value of n_lambdas
        concate_rttm_file = diarize_dataset(
            full_csv, split_type, n_lambdas, pval, nn
        )

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_synd_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        DER_list.append([nn, DER_])

    DER_list.sort(key=lambda x: x[1])
    tunned_nn = DER_list[0]

    return tunned_nn[0]


# Begin!
if __name__ == "__main__":  # noqa: C901

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])

    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from synd_prepare import prepare_synd  # noqa

    run_on_main(
        prepare_synd,
        kwargs={
            "data_folder": params["data_folder"],
            "save_folder": params["save_folder"],
            "max_subseg_dur": params["max_subseg_dur"],
            "overlap": params["overlap"],
        },
    )

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Few more experiment directories (to have cleaner structure)
    exp_dirs = [
        params["embedding_dir"],
        params["csv_dir"],
        params["ref_rttm_dir"],
        params["sys_rttm_dir"],
        params["der_dir"],
    ]
    for dir_ in exp_dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected()
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    # Synd Dev Set
    full_csv = []
    with open(params["csv_diary_dev"], "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_csv.append(row)
       
    best_nn = None
    if params["affinity"] == "nn":
        logger.info("Tuning for nn (Multiple iterations over SynDetect Dev set)")
        best_nn = dev_nn_tuner(full_csv, "dev")
    
    n_lambdas = None
    best_pval = None
    if params["affinity"] == "cos":  # oracle num_spkrs or not doesn't matter
        # cos: Tune for best pval
        logger.info("Tuning for p-value (Multiple iterations over SynDetect Dev set)")
        best_pval = dev_p_tuner(full_csv, "dev")

    # Running once more of dev set (optional)
    out_boundaries = diarize_dataset(
        full_csv,
        "dev",
        n_lambdas=n_lambdas,
        pval=best_pval,
        n_neighbors=best_nn,
    )

    # Evaluating on DEV set
    logger.info("Evaluating for SynDetect Dev. set")
    ref_rttm_dev = os.path.join(params["ref_rttm_dir"], "fullref_synd_dev.rttm")
    sys_rttm_dev = out_boundaries
    [MS_dev, FA_dev, SER_dev, DER_dev] = DER(
        ref_rttm_dev,
        sys_rttm_dev,
        params["ignore_overlap"],
        params["forgiveness_collar"],
        individual_file_scores=True,
    )
    msg = "SynDetect Dev set: Diarization Error Rate = %s %%\n" % (
        str(round(DER_dev[-1], 2))
    )
    logger.info(msg)

    
    # SynDetect Test Set
    full_csv = []
    with open(params["csv_diary_test"], "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            full_csv.append(row)

    out_boundaries = diarize_dataset(
        full_csv,
        "test",
        n_lambdas=n_lambdas,
        pval=best_pval,
        n_neighbors=best_nn,
    )

    # Evaluating on test set
    logger.info("Evaluating for SynDetect test set")
    ref_rttm_test = os.path.join(
        params["ref_rttm_dir"], "fullref_synd_eval.rttm"
    )
    sys_rttm_test = out_boundaries
    [MS_test, FA_test, SER_test, DER_test] = DER(
        ref_rttm_test,
        sys_rttm_test,
        params["ignore_overlap"],
        params["forgiveness_collar"],
        individual_file_scores=True,
    )
    msg = "SynDetect test set: Diarization Error Rate = %s %%\n" % (
        str(round(DER_test[-1], 2))
    )
    logger.info(msg)

    msg = (
        "Final Diarization Error Rate (%%) on SynDetect corpus: Dev = %s %% | Test = %s %%\n"
        % (str(round(DER_dev[-1], 2)), str(round(DER_test[-1], 2)))
    )
    logger.info(msg)

    # Writing der for individual files
    t0 = "oracle" if params["oracle_n_spkrs"] else "est"
    tag = t0 + "_" + str(params["affinity"]) + ".txt"

    out_der_file = os.path.join(params["der_dir"], "dev_DER_" + tag)
    diar.write_ders_file(ref_rttm_dev, DER_dev, out_der_file)

    out_der_file = os.path.join(params["der_dir"], "test_DER_" + tag)
    diar.write_ders_file(ref_rttm_test, DER_test, out_der_file)
