import argparse
from pathlib import Path
from typing import Iterator, Union

import evaluate
from tqdm import tqdm
from whisper.utils import write_srt

from create_data import DataProcessor


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics")
    parser.add_argument(
        "--recognized-dir",
        type=str,
        required=True,
        help="Path to directory containing recognized transcripts in SRT format",
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        required=True,
        help=(
            "Path to directory containing transcripts in SRT format. The filenames under this "
            "directory must match the filenames under `--recognized-dir` directory."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="WER",
        choices=["WER", "CER"],
        help="Evaluation metric",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print out the evaluation results of each file"
    )
    return parser


def srt_to_text(path: Union[str, Path]) -> str:
    utterances = DataProcessor.read_utterances_from_srt(path, normalize_unicode=True)
    return " ".join([u.text for u in utterances])


def save_srt(transcript: Iterator[dict], path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        write_srt(transcript, file=f)


def main():
    args = get_parser().parse_args()

    reference_texts, recognized_texts = [], []
    evaluator = evaluate.load(args.metric.lower())
    score_sum = 0

    for recognized_path in tqdm(list(Path(args.recognized_dir).iterdir())):
        speech_id = Path(recognized_path).stem
        transcript_path = Path(args.transcript_dir) / f"{speech_id}.srt"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        reference_text = srt_to_text(transcript_path)
        recognized_text = srt_to_text(recognized_path)
        reference_texts.append(reference_text)
        recognized_texts.append(recognized_text)

        score = evaluator.compute(references=[reference_text], predictions=[recognized_text])
        if args.verbose:
            tqdm.write(f"Processing: {recognized_path}")
            tqdm.write(f"    {args.metric}: {score}")
        score_sum += score

    print(f"Unweighted Average {args.metric}: {score_sum / len(reference_texts)}")
    weighted_average = evaluator.compute(references=reference_texts, predictions=recognized_texts)
    print(f"Weighted Average {args.metric}: {weighted_average}")


if __name__ == "__main__":
    main()