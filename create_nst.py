import argparse
import json
import os
import random
import unicodedata

import pandas as pd
from scipy.io.wavfile import write
from whisper.tokenizer import get_tokenizer

from create_data import DataProcessor, Record
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/work3/s183954/NST_dk/")
    parser.add_argument("--language", type=str, default="da")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="multilingual",
        choices=["multilingual", "english"],
    )
    parser.add_argument("--normalize_unicode", type=bool, default=False)
    parser.add_argument("--max_tokens_length", type=int, default=219)
    return parser.parse_args()


def main(args):
    args = parse_args()

    # read error files.json
    error_files_path = args.path + "supplement_dk/dk_errorfiles_train.json"
    with open(error_files_path) as file:
        errors = json.load(file)
    df_errors = pd.DataFrame(errors["filesList"])
    df_errors["filepath"] = df_errors["filepath"].apply(lambda x: x[21:])

    if args.split == "train":
        df = pd.read_csv(args.path + "NST_dk_clean.csv", sep=",", low_memory=False)
        path = args.path + "dk/"
        file_names = df["filename_both_channels"]
    elif args.split == "test":
        df = pd.read_csv(args.path + "supplement_dk_clean.csv", sep=",", low_memory=False)
        path = args.path + "supplement_dk/testdata/audio/"
        file_names = df["filename_channel_1"]
    else:
        raise ValueError("split must be either train or test")

    print(f"Processing {args.split} data")
    text_list = df["text"]
    tokenizer = get_tokenizer(multilingual=(args.tokenizer_type == "multilingual"))
    print("processing records")
    records = []
    for item in range(df.shape[0]):
        text = text_list[item]
        filename = file_names[item]
        if type(filename) != str:
            continue
        folder = filename.split("_")[0]

        if args.split != "train":
            filename = filename.split("_")[1]
        else:
            # we need to check if the file is in the error list
            audio_path = f"dk/{folder}/{filename.lower()}"
            if audio_path in df_errors["filepath"].values:
                print(f"Skipping {audio_path} because it is in the error list")
                continue

        auido_path = f"{path}{folder}/{filename.lower()}"

        if args.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        tokens = tokenizer.encode(text)
        if len(tokens) > args.max_tokens_length:
            print(
                f"Skipping {path} ({text}) because it is too long "
                f"({len(tokens)} tokens)"
            )
            continue
        record = Record(audio_path=auido_path, text=text, language=args.language)
        records.append(record)

    print(f"Saving {len(records)}")
    if args.split == "train":
        # use 10 % of the training data for validation
        random.shuffle(records)
        train_records = records[: int(len(records) * 0.9)]
        val_records = records[int(len(records) * 0.9) :]
        DataProcessor.write_records(train_records, "nst_train.json")
        DataProcessor.write_records(val_records, "nst_dev.json")
    else:
        DataProcessor.write_records(records, "nst_test.json")


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
