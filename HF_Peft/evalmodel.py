import argparse
import gc

import evaluate
import numpy as np
import torch
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from dataset import DataCollatorSpeechSeq2SeqWithPadding, get_dataset_eval


def arrguement_parser():
    # hf model
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model", type=str, help="huggingface model")
    parser.add_argument("--hf_finetuned", type=str, help="huggingface model")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument(
        "--task", type=str, help="task to perform", default="transcribe"
    )

    return parser.parse_args()


def main(args):
    language = "Danish"
    peft_config = PeftConfig.from_pretrained(args.hf_finetuned)

    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, device_map="auto"
    )

    model = PeftModel.from_pretrained(model, args.hf_finetuned)
    model.config.use_cache = True
    tokenizer = WhisperTokenizer.from_pretrained(
        args.hf_model, language=language, task=args.task
    )
    processor = WhisperProcessor.from_pretrained(
        args.hf_model, language=language, task=args.task
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.hf_model)
    common_voice = get_dataset_eval(feature_extractor, tokenizer)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    eval_dataloader = DataLoader(
        common_voice["test"], batch_size=args.batch_size, collate_fn=data_collator
    )
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task=args.task
    )
    normalizer = BasicTextNormalizer()

    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []
    metric = evaluate.load("wer")
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(
                    labels != -100, labels, processor.tokenizer.pad_token_id
                )
                decoded_preds = processor.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = processor.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                normalized_predictions.extend(
                    [normalizer(pred).strip() for pred in decoded_preds]
                )
                normalized_references.extend(
                    [normalizer(label).strip() for label in decoded_labels]
                )
            del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * metric.compute(predictions=predictions, references=references)
    normalized_wer = 100 * metric.compute(
        predictions=normalized_predictions, references=normalized_references
    )
    eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

    print(f"{wer=} and {normalized_wer=}")
    print(eval_metrics)


if __name__ == "__main__":
    args = arrguement_parser()
    main(args)
