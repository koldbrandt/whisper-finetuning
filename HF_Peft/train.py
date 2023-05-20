import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from peft import LoraConfig, LoraModel, PeftModel, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from datasets import Audio, DatasetDict, load_dataset


def arrguement_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, help="huggingface token")
    parser.add_argument("--hf_model", type=str, help="huggingface model")
    parser.add_argument(
        "--task", type=str, help="task to perform", default="transcribe"
    )
    parser.add_argument(
        "--cache_dir", type=str, help="cache directory", default="/work3/s183954/"
    )
    parser.add_argument(
        "--output_hf_model",
        type=str,
        help="output huggingface model",
        default=f"koldborg/{args.hf_model.split('/')[1]}-peft",
    )
    return parser.parse_args()


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


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


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


def main(args):
    dataset_name = "mozilla-foundation/common_voice_13_0"
    language = "Danish"
    language_abbr = "da"
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset(
        dataset_name,
        language_abbr,
        split="train+validation",
        use_auth_token=True,
        data_dir="/work3/s183954/",
        cache_dir="/work3/s183954/",
    )
    common_voice["test"] = load_dataset(
        dataset_name,
        language_abbr,
        split="test",
        use_auth_token=True,
        data_dir="/work3/s183954/",
        cache_dir="/work3/s183954/",
    )
    common_voice = common_voice.remove_columns(
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

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.hf_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.hf_model, language=language, task=args.task
    )
    processor = WhisperProcessor.from_pretrained(
        args.hf_model, language=language, task=args.task
    )

    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice.map(
        prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained(
        args.hf_model, cache_dir=args.cache_dir
    )
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    training_args = Seq2SeqTrainingArguments(
        output_dir="reach-vb/test",  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=1,
        evaluation_strategy="steps",
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=100,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()


if __name__ == "__main__":
    print("Hello World")
    args = arrguement_parser()
    main(args)
