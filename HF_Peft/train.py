import argparse
import os

import evaluate
import torch
from dataset import DataCollatorSpeechSeq2SeqWithPadding, get_dataset_train_test
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
        "--hf_username", type=str, help="huggingface username", default="koldborg"
    )
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--num_proc", type=int, help="number of processes", default=8)
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

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    def preprocess_logits_for_metrics(logits, labels):

        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels


    language = "Danish"

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.hf_model, cache_dir=args.cache_dir )
    tokenizer = WhisperTokenizer.from_pretrained(
        args.hf_model, language=language, task=args.task, use_auth_token=args.hf_token, cache_dir=args.cache_dir
    )
    processor = WhisperProcessor.from_pretrained(
        args.hf_model, language=language, task=args.task, cache_dir=args.cache_dir, use_auth_token=args.hf_token
    )

    datasets = get_dataset_train_test(
        args.cache_dir,
        args.cache_dir,
        feature_extractor,
        tokenizer,
        ["common_voice", "fleurs"],
        args.hf_token,
        args.num_proc,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    model = WhisperForConditionalGeneration.from_pretrained(
        args.hf_model, cache_dir=args.cache_dir,device_map="auto"
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

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
        gradient_accumulation_steps=1,  
        learning_rate=2e-4,
        warmup_steps=50,
        num_train_epochs=3,
        evaluation_strategy="steps",
        fp16=True,
        per_device_eval_batch_size=args.batch_size,
        generation_max_length=128,
        logging_steps=50,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
        logging_first_step=True,
        load_best_model_at_end=True,
        # metric_for_best_model="wer",
        # greater_is_better=False,
        # eval_accumulation_steps=1,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.evaluate()
    trainer.train()

    peft_model_id = f"{args.hf_username}/{args.hf_model.split('/')[-1]}-{language}"
    model.push_to_hub(peft_model_id)
    print(f"Pushed to {peft_model_id}")



if __name__ == "__main__":
    args = arrguement_parser()
    main(args)
