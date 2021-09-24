"""
Downloads and creates data manifest files for Mini LibriSpeech (spk-id).
For speaker-id, different sentences of the same speaker must appear in train,
validation, and test sets. In this case, these sets are thus derived from
splitting the original training set intothree chunks.

Authors:
 * Mirco Ravanelli, 2021
"""

import os
import json
from tqdm import tqdm
import logging
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
data_folder = "/home/bowen/data/ASVspoof2019_root/LA/"
SAMPLERATE = 16000


def prepare_asv_spoof(
    data_folder,
    save_json_train,
    save_json_dev,
    save_json_eval,
):
    """
    Prepares the json files for the Mini Librispeech dataset.

    Downloads the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Example
    -------
    >>> data_folder = "/home/bowen/data/ASVspoof2019_root/LA/"
    >>> prepare_asv_spoof(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_dev, save_json_eval):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # data folder and annotation folder
    train_folder = os.path.join(data_folder, "ASVspoof2019_LA_train", "flac")
    dev_folder = os.path.join(data_folder, "ASVspoof2019_LA_dev", "flac")
    eval_folder = os.path.join(data_folder, "ASVspoof2019_LA_eval", "flac")
    annotation_folder = os.path.join(data_folder, "ASVspoof2019_LA_cm_protocols")

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_dev}, and {save_json_eval}"
    )
    extension = [".flac"]
    tran_wav_list = get_all_files(train_folder, match_and=extension)
    dev_wav_list = get_all_files(dev_folder, match_and=extension)
    eval_wav_list = get_all_files(eval_folder, match_and=extension)

    # Read ground truth from annotation files
    train_anno = []
    with open(annotation_folder+'/ASVspoof2019.LA.cm.train.trn.txt') as file:
        for line in file:
            line = line.strip()
            train_anno.append(line.split(' '))
    dev_anno = []
    with open(annotation_folder+'/ASVspoof2019.LA.cm.dev.trl.txt') as file:
        for line in file:
            line = line.strip()
            dev_anno.append(line.split(' '))
    eval_anno = []
    with open(annotation_folder+'/ASVspoof2019.LA.cm.eval.trl.txt') as file:
        for line in file:
            line = line.strip()
            eval_anno.append(line.split(' '))

    # Creating json files
    create_json(tran_wav_list, train_anno, train_folder, save_json_train)
    create_json(dev_wav_list, dev_anno, dev_folder, save_json_dev)
    create_json(eval_wav_list, eval_anno, eval_folder, save_json_eval)


def create_json(wav_list, anno_list, root, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    anno_list : list of str
        The list of annotations
    root: str
        The path of the audio files
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}
    wav_list.sort()
    anno_list.sort(key=lambda x: x[1])
    if len(wav_list) == len(anno_list):
        for i, wav_file in tqdm(enumerate(wav_list)):

            # Reading the signal (to retrieve duration in seconds)
            signal = read_audio(wav_file)
            duration = signal.shape[0] / SAMPLERATE

            # Getting key from annotation
            key = anno_list[i][-1]
            spk_id = anno_list[i][0]
            filename = anno_list[i][1]

            # Create entry for this utterance
            json_dict[filename] = {
                "path": wav_file,
                "length": duration,
                "type": key,
                "speaker_id": spk_id,
            }
    else:
        for i, wav_file in tqdm(enumerate(anno_list)):
            key = anno_list[i][-1]
            spk_id = anno_list[i][0]
            filename = anno_list[i][1]
            wav_file = os.path.join(root, filename+".flac")

            # Reading the signal (to retrieve duration in seconds)
            if os.path.isfile(wav_file):
                signal = read_audio(wav_file)
                duration = signal.shape[0] / SAMPLERATE
            else:
                with open("wrong anno.csv", "a") as f:
                    f.write(wav_file + ",no wav file\n")
                continue

            # Create entry for this utterance
            json_dict[filename] = {
                "path": wav_file,
                "length": duration,
                "type": key,
                "speaker_id": spk_id,
            }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True



prepare_asv_spoof(data_folder, 'train.json', 'dev.json', 'eval.json')
