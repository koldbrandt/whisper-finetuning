import argparse
import os

import torch
from datasets import Audio, DatasetDict, load_dataset
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

from dataset import DataCollatorSpeechSeq2SeqWithPadding, get_dataset_train_test


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
        "--hf_username", type=str, help="huggingface username", default="koldborg"
    )
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--num_proc", type=int, help="number of processes", default=4)
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


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


def main(args):
    language = "Danish"

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.hf_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.hf_model, language=language, task=args.task
    )
    processor = WhisperProcessor.from_pretrained(
        args.hf_model, language=language, task=args.task
    )

    datasets = get_dataset_train_test(
        args.cache_dir, args.cache_dir, feature_extractor, tokenizer,["common_voice", "fleurs"] ,args.num_proc
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

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

    output_dir = f"{args.hf_username}/test"

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=1,
        evaluation_strategy="steps",
        fp16=True,
        per_device_eval_batch_size=args.batch_size,
        generation_max_length=128,
        logging_steps=100,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()

    peft_model_id = f"{args.hf_username}/{args.hf_model.split('/')[-1]}-{language}"
    model.push_to_hub(peft_model_id)
    print(f"Pushed to {peft_model_id}")


if __name__ == "__main__":
    args = arrguement_parser()
    main(args)
