from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import Audio, DatasetDict, load_dataset, concatenate_datasets, Dataset

language = "Danish"


dataset_dict = {
    "common_voice": {
        "name": "mozilla-foundation/common_voice_13_0",
        "language": "Danish",
        "language_abbr": "da",
    },
    "fleurs": {"name": "google/fleurs", "language": "Danish", "language_abbr": "da_dk"},
}


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


def remove_fleurs_columns(dataset: DatasetDict):
    dataset = dataset.remove_columns(
        [
            "id",
            "num_samples",
            "path",
            "transcription",
            "gender",
            "lang_id",
            "language",
            "lang_group_id",
        ]
    )
    dataset = dataset.rename_column("raw_transcription", "sentence")
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


def get_dataset_dict_test(
    dataset_name: str, datadir: str, cachedir: str, datasetdict: DatasetDict
):
    temp_dataset = DatasetDict()
    temp_dataset["test"] = load_dataset(
        dataset_dict[dataset_name]["name"],
        dataset_dict[dataset_name]["language_abbr"],
        split="test",
        use_auth_token=True,
        data_dir=datadir,
        cache_dir=cachedir,
    )

    if dataset_name == "common_voice":
        temp_dataset = remove_commonvoice_columns(temp_dataset)
    elif dataset_name == "fleurs":
        temp_dataset = remove_fleurs_columns(temp_dataset)

    temp_dataset = temp_dataset.cast_column("audio", Audio(sampling_rate=16000))

    if "test" in datasetdict.keys():
        datasetdict["test"] = concatenate_datasets(
            [datasetdict["test"], temp_dataset["test"]]
        )
    else:
        datasetdict["test"] = temp_dataset["test"]

    return datasetdict


def get_dataset_eval(datadir, cachedir,feature_extractor, tokenizer, dataset_name, num_proc=4):
    datasets = DatasetDict()

    for dataset in dataset_name:
        datasets = get_dataset_dict_test(dataset, datadir, cachedir, datasets)

    datasets = datasets.map(
        prepare_dataset,
        num_proc=num_proc,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
    )

    return datasets


def get_dataset_dict_train_test(
    dataset_name: str, datadir: str, cachedir: str, datasetdict: DatasetDict
):
    temp_dataset = DatasetDict()
    temp_dataset["train"] = load_dataset(
        dataset_dict[dataset_name]["name"],
        dataset_dict[dataset_name]["language_abbr"],
        split="train",
        use_auth_token=True,
        data_dir=datadir,
        cache_dir=cachedir,
    )
    temp_dataset["test"] = load_dataset(
        dataset_dict[dataset_name]["name"],
        dataset_dict[dataset_name]["language_abbr"],
        split="validation",
        use_auth_token=True,
        data_dir=datadir,
        cache_dir=cachedir,
    )
    if dataset_name == "common_voice":
        temp_dataset = remove_commonvoice_columns(temp_dataset)
    elif dataset_name == "fleurs":
        temp_dataset = remove_fleurs_columns(temp_dataset)

    temp_dataset = temp_dataset.cast_column("audio", Audio(sampling_rate=16000))

    if "train" in datasetdict.keys() and "test" in datasetdict.keys():
        datasetdict["train"] = concatenate_datasets(
            [datasetdict["train"], temp_dataset["train"]]
        )
        datasetdict["test"] = concatenate_datasets(
            [datasetdict["test"], temp_dataset["test"]]
        )
    else:
        datasetdict["train"] = temp_dataset["train"]
        datasetdict["test"] = temp_dataset["test"]

    return datasetdict


def get_dataset_train_test(
    datadir,
    cachedir,
    feature_extractor,
    tokenizer,
    dataset_name: Union[str, list],
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

    if type(dataset_name) != list:
        dataset_name = [dataset_name]

    datasets = DatasetDict()

    for dataset in dataset_name:
        datasets = get_dataset_dict_train_test(dataset, datadir, cachedir, datasets)

    datasets = datasets.map(
        prepare_dataset,
        num_proc=num_proc,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
    )

    return datasets
