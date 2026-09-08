import glob
from pathlib import Path
import pandas as pd
import numpy as np

def load_metadata(data_dir="free-spoken-digit-dataset/recordings"):
    """Parses FSDD filenames for digits and speakers."""
    files = sorted(glob.glob(f"{data_dir}/*.wav"))
    records = []
    for f in files:
        stem = Path(f).stem
        digit_str, speaker, idx = stem.split("_")
        records.append({"path": f, "digit": int(digit_str), "speaker": speaker, "idx": int(idx)})
    return pd.DataFrame(records)

def create_grouped_split(meta_df, test_speaker="yweweler"):
    """
    Creates a leakage-free train/test split.
    Holds out exactly one speaker ('yweweler') for test, leaving exactly five for train.
    """
    train_mask = meta_df["speaker"] != test_speaker
    test_mask = ~train_mask
    return train_mask.to_numpy(), test_mask.to_numpy()