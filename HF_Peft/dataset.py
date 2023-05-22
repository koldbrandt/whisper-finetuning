from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import Audio, DatasetDict, load_dataset

dataset_name = "mozilla-foundation/common_voice_13_0"
language = "Danish"
language_abbr = "da"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def remove_commonvoice_columns(dataset):
    dataset = dataset.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
            "variant",
        ]
    )
    return dataset


def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def get_dataset_eval(feature_extractor, tokenizer, num_proc=4):
    common_voice = DatasetDict()

    common_voice["test"] = load_dataset(
        dataset_name,
        language_abbr,
        split="test",
        use_auth_token=True,
        data_dir="/work3/s183954/",
        cache_dir="/work3/s183954/",
    )

    common_voice = remove_commonvoice_columns(common_voice)
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    common_voice = common_voice.map(
        prepare_dataset,
        num_proc=num_proc,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
    )

    return common_voice


def get_dataset_train_test(
    datadir,
    cachedir,
    feature_extractor,
    tokenizer,
    num_proc=4,
):
    """
    Load and prepare Common Voice dataset for training and testing.

    Args:
        datadir (str): Path to dataset directory.
        cachedir (str): Path to cache directory.
        feature_extractor (WhisperFeatureExtractor): Feature extractor for audio data.
        tokenizer (WhisperTokenizer): Tokenizer for text data.
        num_proc (int): Number of processes to use for data preprocessing.

    Returns:
        common_voice (DatasetDict): DatasetDict containing train and test split.
    """
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset(
        dataset_name,
        language_abbr,
        split="train",
        use_auth_token=True,
        data_dir=datadir,
        cache_dir=cachedir,
    )
    common_voice["test"] = load_dataset(
        dataset_name,
        language_abbr,
        split="validation",
        use_auth_token=True,
        data_dir=datadir,
        cache_dir=cachedir,
    )

    common_voice = remove_commonvoice_columns(common_voice)

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],
        num_proc=num_proc,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
    )

    return common_voice
