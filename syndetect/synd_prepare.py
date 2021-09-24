import os
import logging
import csv
import pandas as pd
import xml.etree.ElementTree as et
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_synd_prepare.pkl"
TRAIN_CSV = "synd_train.subsegments.csv"
DEV_CSV = "synd_dev.subsegments.csv"
TEST_CSV = "synd_test.subsegments.csv"
SAMPLERATE = 24000


def prepare_synd(
    data_folder,
    save_folder,
    max_subseg_dur=3.0,
    overlap=1.5,
):
    """
    Prepares reference RTTM and CSV files for the SynDetect dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the SynDetect wavfiles and annotations are stored.
    save_folder : str
        The save directory in results.
    max_subseg_dur : float
        Duration in seconds of a subsegments to be prepared from larger segments.
    overlap : float
        Overlap duration in seconds between adjacent subsegments

    Example
    -------
    >>> from workdir.syndetect.syndetect_prepare import prepare_synd
    >>> data_folder = '/home/zhangbowen/data2/SynDetect_gutenburg/'
    >>> save_folder = '/home/zhangbowen/data2/results/save/'
    >>> prepare_synd(data_folder, save_folder)
    """

    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "save_folder": save_folder,
        "max_subseg_dur": max_subseg_dur,
        "overlap": overlap,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Setting ouput opt files
    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    splits = ["train", "dev", "test"]
    if skip(splits, save_folder, conf):
        logger.info(
            "Skipping data preparation, as it was completed in previous run."
        )
        return

    msg = "\tCreating csv file for the SynDetect Dataset.."
    logger.debug(msg)

    # Prepare RTTM from csv(manual annot) and store are groundtruth
    # Create ref_RTTM directory
    ref_dir = save_folder + "/ref_rttms/"
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)

    # Create reference RTTM files
    for i in splits:
        rttm_file = ref_dir + "/fullref_synd_" + i + ".rttm"
        prepare_segs_for_RTTM(data_folder, i, rttm_file)
        
    # Create csv_files for splits
    csv_folder = os.path.join(save_folder, "csv")
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        
    print("Creating csv file for the SynDetect Dataset")
    for i in splits:
        rttm_file = ref_dir + "/fullref_synd_" + i + ".rttm"
        csv_filename_prefix = "synd_" + i
        prepare_csv(
            rttm_file,
            csv_folder,
            data_folder,
            csv_filename_prefix,
            max_subseg_dur,
            overlap,
            i
        )
        print(i + "completed")

    save_pkl(conf, save_opt)


def get_RTTM_per_rec(segs, spkrs_list, rec_id):
    """Prepares rttm for each recording
    """

    rttm = []

    # Prepare header
    for spkr_id in spkrs_list:
        # e.g. SPKR-INFO ES2008c 0 <NA> <NA> <NA> unknown ES2008c.A_PM <NA> <NA>
        line = (
            "SPKR-INFO "
            + rec_id
            + " 0 <NA> <NA> <NA> unknown "
            + spkr_id
            + " <NA> <NA>"
        )
        rttm.append(line)

    # Append remaining lines
    for row in segs:
        # e.g. SPEAKER ES2008c 0 37.880 0.590 <NA> <NA> ES2008c.A_PM <NA> <NA>

        if float(row[1]) < float(row[0]):
            msg1 = (
                "Possibly Incorrect Annotation Found!! transcriber_start (%s) > transcriber_end (%s)"
                % (row[0], row[1])
            )
            msg2 = (
                "Excluding this incorrect row from the RTTM : %s, %s, %s, %s"
                % (
                    rec_id,
                    row[0],
                    str(round(float(row[1]) - float(row[0]), 4)),
                    str(row[2]),
                )
            )
            logger.info(msg1)
            logger.info(msg2)
            continue

        line = (
            "SPEAKER "
            + rec_id
            + " 0 "
            + str(round(float(row[0]), 4))
            + " "
            + str(round(float(row[1]) - float(row[0]), 4))
            + " <NA> <NA> "
            + str(row[2])
            + " <NA> <NA>"
        )
        rttm.append(line)

    return rttm


def prepare_segs_for_RTTM(data_folder, split, out_rttm_file):
    RTTM = []  # Stores all RTTMs clubbed together for a given dataset split
    df = pd.read_csv(data_folder+split+"_annotation.csv", header=None)
    df.columns = ["uttid", "t1", "t2", "t3"]
    annos = df.values
 
    for row in range(annos.shape[0]): 
        segs = []
        spkrs_list = ([])
        uttid = annos[row][0]
        t1 = annos[row][1]
        t2 = annos[row][2]
        t3 = annos[row][3]
        spkrs_list.append(uttid + ".A")
        spkrs_list.append(uttid + ".B")

        # Start, end and speaker_ID from xml file
        segs = segs + [
            [0, t1, uttid +'.A'],
            [t1, t2, uttid +'.B'],
            [t2, t3, uttid +'.A'],
        ]

        rttm_per_rec = get_RTTM_per_rec(segs, spkrs_list, uttid)
        RTTM = RTTM + rttm_per_rec

    # Write one RTTM as groundtruth. For example, "fullref_test.rttm"
    with open(out_rttm_file, "w") as f:
        for item in RTTM:
            f.write("%s\n" % item)

        
def get_subsegments(big_segs, max_subseg_dur=3.0, overlap=1.5):
    """Divides bigger segments into smaller sub-segments
    """

    shift = max_subseg_dur - overlap
    subsegments = []

    # These rows are in RTTM format
    for row in big_segs:
        seg_dur = float(row[4])
        rec_id = row[1]

        if seg_dur > max_subseg_dur:
            num_subsegs = int(seg_dur / shift)
            # Taking 0.01 sec as small step
            seg_start = float(row[3])
            seg_end = seg_start + seg_dur

            # Now divide this segment (new_row) in smaller subsegments
            for i in range(num_subsegs):
                subseg_start = seg_start + i * shift
                subseg_end = min(subseg_start + max_subseg_dur - 0.01, seg_end)
                subseg_dur = subseg_end - subseg_start

                new_row = [
                    "SPEAKER",
                    rec_id,
                    "0",
                    str(round(float(subseg_start), 4)),
                    str(round(float(subseg_dur), 4)),
                    "<NA>",
                    "<NA>",
                    row[7],
                    "<NA>",
                    "<NA>",
                ]

                subsegments.append(new_row)

                # Break if exceeding the boundary
                if subseg_end >= seg_end:
                    break
        else:
            subsegments.append(row)

    return subsegments
    

def prepare_csv(
    rttm_file, save_dir, data_dir, filename, max_subseg_dur, overlap, split
):
    # Read RTTM, get unique meeting_IDs (from RTTM headers)
    # For each MeetingID. select that meetID -> merge -> subsegment -> csv -> append

    # Read RTTM
    RTTM = []
    with open(rttm_file, "r") as f:
        for line in f:
            entry = line[:-1]
            RTTM.append(entry)

    spkr_info = filter(lambda x: x.startswith("SPKR-INFO"), RTTM)
    rec_ids = list(set([row.split(" ")[1] for row in spkr_info]))
    rec_ids.sort()  # sorting just to make CSV look in proper sequence

    # For each recording merge segments and then perform subsegmentation
    ORIGION_SEGMENTS = []
    SUBSEGMENTS = []
    for rec_id in rec_ids:
        segs_iter = filter(
            lambda x: x.startswith("SPEAKER " + str(rec_id)), RTTM
        )
        gt_rttm_segs = [row.split(" ") for row in segs_iter]

        ORIGION_SEGMENTS = ORIGION_SEGMENTS + gt_rttm_segs

        # Divide segments into smaller sub-segments
        subsegs = get_subsegments(gt_rttm_segs, max_subseg_dur, overlap)
        SUBSEGMENTS = SUBSEGMENTS + subsegs

    # Write segment AND sub-segments (in RTTM format)
    segs_file = save_dir + "/" + filename + ".segments.rttm"
    subsegment_file = save_dir + "/" + filename + ".subsegments.rttm"

    with open(segs_file, "w") as f:
        for row in ORIGION_SEGMENTS:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)

    with open(subsegment_file, "w") as f:
        for row in SUBSEGMENTS:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)

    # Create CSV from subsegments
    csv_output_head = [["ID", "duration", "wav", "start", "stop"]]  # noqa E231

    entry = []
    for row in SUBSEGMENTS:
        rec_id = row[1]
        strt = str(round(float(row[3]), 4))
        end = str(round((float(row[3]) + float(row[4])), 4))

        subsegment_ID = rec_id + "_" + strt + "_" + end
        dur = row[4]
        wav_file_path = (
            data_dir
            + split
            + "/"
            + rec_id
            + ".wav"
        )

        start_sample = int(float(strt) * SAMPLERATE)
        end_sample = int(float(end) * SAMPLERATE)

        # Composition of the csv_line
        csv_line = [
            subsegment_ID,
            dur,
            wav_file_path,
            str(start_sample),
            str(end_sample),
        ]

        entry.append(csv_line)

    csv_output = csv_output_head + entry

    # Write csv file only for subsegments
    csv_file = save_dir + "/" + filename + ".subsegments.csv"
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    msg = "%s csv prepared" % (csv_file)
    logger.debug(msg)
    

def skip(splits, save_folder, conf):
    """
    Detects if the data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
        "test": TEST_CSV,
    }
    for split in splits:
        if not os.path.isfile(
            os.path.join(save_folder, "csv", split_files[split])
        ):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip

# if __name__ == "__main__":
#     data_folder = '/home/zhangbowen/data2/SynDetect_gutenburg/'
#     save_folder = '/home/zhangbowen/data2/results/save/'
#     prepare_synd(data_folder, save_folder)