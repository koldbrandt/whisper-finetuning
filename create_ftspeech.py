import argparse
import pandas as pd
import torchaudio
from tqdm import tqdm
from create_data import DataProcessor, Record
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/dtu/blackhole/1f/137151/ftspeech")
    parser.add_argument("--output_dir", type=str, default="/dtu/blackhole/1f/137151/ftspeech_clean")
    parser.add_argument("--data_file", type=str, default="ft-speech_train.tsv")
    parser.add_argument("--remove_long", type=bool, default=True)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--num_processes", type=int, default=4)  # Number of processes to use
    return parser.parse_args()


def process_chunk(args, chunk_index, df_chunk):
    last_filepath = ""
    records = []

    with tqdm(total=df_chunk.shape[0], desc=f"Chunk {chunk_index+1}") as pbar:
        for index, row in df_chunk.iterrows():
            folder = row['utterance_id'].split('_')[1][:-1]
            filename = f"{row['utterance_id'][5:15]}.wav"
            filepath = f'{args.data_dir}/audio/{folder}/{filename}'

            if filepath != last_filepath:
                last_filepath = filepath
                auido, sr = torchaudio.load(filepath)

            start = int(row['start_time'] * sr)
            end = int(row['end_time'] * sr)
            audio_temp = auido[:, start:end]

            savepath = f'{args.output_dir}/audio/{row["utterance_id"]}.wav'
            torchaudio.save(savepath, audio_temp, sr)

            text = row['transcript'].strip()
            text = text.replace("<UNK>", "")
            record = Record(
                audio_path=savepath,
                text=text,
                language="da",
                prompt="",
            )
            records.append(record)
            pbar.update(1)

    return records


def main(args):
    path_data = f"{args.data_dir}/text/{args.data_file}"
    df = pd.read_csv(path_data, sep="\t")
    if args.remove_long:
        df["duration"] = df["end_time"] - df["start_time"]
        df = df[df["duration"] <= 30]
    df = df.reset_index(drop=True)
    if args.max_utterances:
        df = df.iloc[: args.max_utterances]

    chunk_size = len(df) // args.num_processes
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    records = []

    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        futures = []
        chunk_indices = range(len(chunks))

        for chunk_index, chunk in zip(chunk_indices, chunks):
            future = executor.submit(process_chunk, args, chunk_index, chunk)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing Chunks'):
            records.extend(future.result())

    # save records
    DataProcessor.write_records(records, f"{args.output_dir}/data/{args.data_file[:-4]}.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)

