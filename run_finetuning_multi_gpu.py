import argparse
import copy
import glob
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import whisper
from torch.distributed import destroy_process_group, init_process_group, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from whisper import Whisper
from whisper.tokenizer import Tokenizer, get_tokenizer

# from torch.utils.tensorboard import SummaryWriter
import wandb
from dataloader import collate_fn, get_dataloader, get_dataset
from utilities import calculate_WER, get_normalizer
from Whisper_no_sparse import Whisper_no_sparse

# writer = SummaryWriter()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Whisper",
    entity="fuzzy-fish-waffle",
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    # Dataloader-related arguments
    parser.add_argument(
        "--train-folder",
        type=str,
        required=True,
        help="folder, will look for all json files in the folder",
    )
    parser.add_argument(
        "--dev-folder",
        type=str,
        required=True,
        help="foler, will look for all json files in the folder",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--dev-batch-size", type=int, default=16, help="Batch size for validation"
    )
    parser.add_argument(
        "--no-timestamps-training",
        action="store_true",
        help="Always use the no-timestamps training mode",
    )
    parser.add_argument(
        "--prompt-use-rate",
        type=float,
        default=0.5,
        help="How often to use prompts for conditioning the generation",
    )
    parser.add_argument(
        "--no-timestamps-rate",
        type=float,
        default=0.5,
        help=(
            "How often to use the no-timestamps mode. Only used if --no-timestamps-training "
            "is NOT set"
        ),
    )

    # Training-related arguments
    parser.add_argument(
        "--save-dir", type=str, default="output", help="directory to save the model"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for training",
    )
    parser.add_argument(
        "--model",
        default="large",
        choices=whisper.available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--train-only-decoder", action="store_true", help="train only the decoder"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--accum-grad-steps",
        type=int,
        default=64,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=5000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Number of steps to evaluate the model",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Save all checkpoints instead of only the best and the last one",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use-adam-8bit",
        action="store_true",
        help="Use Adam 8bit optimizer for reduced VRAM usage.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for the dataloader",
    )

    parser.add_argument(
        "--weight-decay", type=float, default=0.1, help="Weight decay for the optimizer"
    )

    parser.add_argument(
        "--adam-eps", type=float, default=1e-6, help="Epsilon for the Adam optimizer"
    )
    return parser



whisper_cache_dir = "/work3/s183954/whisper"

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        dev_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        gpu_id: int,
        tokenizer: Tokenizer,
        normalizer,
        args,
    ) -> None:
        self.model = model.to(f"cuda:{gpu_id}")
        self.train_data = train_data
        self.dev_data = dev_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gpu_id = gpu_id
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.grad_step = args.accum_grad_steps
        self.train_steps = args.train_steps
        self.eval_steps = args.eval_steps
        self.max_grad_norm = args.max_grad_norm
        self.train_only_decoder = args.train_only_decoder
        self.save_dir = args.save_dir
        self.save_all_checkpoints = args.save_all_checkpoints
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _train_step(self):
        self.model.train()
        total_loss = 0
        for _ in range(self.grad_step):
            x, y_in, y_out = next(self.train_data)
            x, y_in, y_out = (
                x.to(self.gpu_id),
                y_in.to(self.gpu_id),
                y_out.to(self.gpu_id),
            )

            if self.train_only_decoder:
                with torch.no_grad():
                    audio_features = self.model.module.embed_audio(x)
            else:
                audio_features = self.model.module.embed_audio(x)
            logits = self.model.module.logits(y_in, audio_features=audio_features)
            loss = F.cross_entropy(logits.transpose(1, 2), y_out)

            loss = loss / self.grad_step
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(
            self.model.module.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return total_loss

    @torch.no_grad()
    def _evaluate(self):
        self.model.module.eval()
        total_loss = 0
        total_wer = 0
        for x, y_in, y_out in tqdm(self.dev_data):
            x, y_in, y_out = (
                x.to(self.gpu_id),
                y_in.to(self.gpu_id),
                y_out.to(self.gpu_id),
            )
            logits = self.model.module(x, y_in)
            loss = F.cross_entropy(logits.transpose(1, 2), y_out)
            wer = calculate_WER(
                logits.argmax(dim=-1), y_out, self.normalizer, self.tokenizer
            )
            total_wer += wer
            total_loss += loss.item()
        return (total_loss / len(self.dev_data)), (total_wer / len(self.dev_data))

    def _save_checkpoint(self, save_path: str):
        ckp = self.model.module.state_dict()
        # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
        torch.save(
            {"model_state_dict": ckp, "dims": asdict(self.model.module.dims)}, save_path
        )

    def train(self):
        if self.gpu_id == 0:
            print("Running initial evaluation... on GPU 0")
            min_loss, init_wer = self._evaluate()
            wandb.log({"Loss/eval": min_loss, "WER/eval": init_wer}, step=0)
            print(f"Initial loss: {min_loss}, Initial WER: {init_wer}")

        barrier()
        print(f"Starting training on GPU {self.gpu_id}")
        for step in range(1, self.train_steps + 1):
            start = time.time()
            train_loss = self._train_step()
            # writer.add_scalar("Loss/train", train_loss, step)
            wandb.log({"Loss/train": train_loss}, step=step)
            end = time.time()
            if step % 10 == 0:
                print(f"Step {step}: training loss={train_loss}, time={end-start}, gpu={self.gpu_id}")
            if step % self.eval_steps == 0:
                if self.gpu_id == 0:
                    eval_loss, eval_wer = self._evaluate()
                    # writer.add_scalar("Loss/eval", eval_loss, step)
                    wandb.log({"Loss/eval": eval_loss, "WER/eval": eval_wer}, step=step)
                    tqdm.write(f"Step {step}: validation loss={eval_loss}")
                    if eval_loss < min_loss:
                        min_loss = eval_loss
                        self._save_checkpoint(f"{self.save_dir}/best_model.pt")
                    if self.save_all_checkpoints:
                        self._save_checkpoint(f"{self.save_dir}/step{step}.pt")

                    self._save_checkpoint(f"{self.save_dir}/last_model.pt")
                barrier()



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))


def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        collate_fn=collate_fn,
    )


def get_dataloaders(args, tokenizer, fp16, max_prompt_length, device):
    train_json = []
    if args.train_folder is not None:
        train_json = glob.glob(os.path.join(args.train_folder, "*"))

    if train_json == []:
        print("No training files found in --train-folder")
        exit(1)
    dataset = get_dataset(
        json=train_json,
        tokenizer=tokenizer,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=args.prompt_use_rate,
        no_timestamps_rate=args.no_timestamps_rate,
    )
    train_loader = prepare_dataloader(dataset, args.batch_size)

    dev_loader = None
    if device == 0:
        dev_json = []
        if args.train_folder is not None:
            dev_json = glob.glob(os.path.join(args.dev_folder, "*"))
        if dev_json == []:
            print("No training files found in --train-folder")
            exit(1)
        print("Build train loader done, with {} batches".format(len(train_loader)))
        dev_loader = get_dataloader(
            json=dev_json,
            tokenizer=tokenizer,
            batch_size=args.dev_batch_size,
            fp16=fp16,
            no_timestamps_training=args.no_timestamps_training,
            max_prompt_length=max_prompt_length,
            # always use prompts and timestamps for validation to make it deterministic
            prompt_use_rate=1.0,
            no_timestamps_rate=0.0,
            shuffle=False,
            workers=0,
        )

    return train_loader, dev_loader


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12366"
    init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )  # Options NCCl, Gloo, MPI
    torch.cuda.set_device(rank)


def load_train_objs(args):
    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    # DDP doesnt support sparse tensors. Newest version of whisper is saving sparse vectors to reister_buffers
    # So we need to load the model and then load it as a whsiper_no_sparse model
    model_with_sparse = whisper.load_model(args.model, device="cpu", download_root=whisper_cache_dir)
    model = Whisper_no_sparse(model_with_sparse.dims).to("cpu")
    model.load_state_dict(model_with_sparse.state_dict())
    return tokenizer, model


def get_optimizer_scheduler(model, args):
    if args.use_adam_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "For using Adam 8bit optimizer you need to have bitsandbytes installed."
            )
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.adam_eps,
            betas=(0.9, 0.98),
        )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )
    return optimizer, scheduler


def main(rank, args, world_size):
    print(f"Running on {rank}")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    ddp_setup(rank, world_size)
    tokenizer, model = load_train_objs(args)
    optimizer, scheduler = get_optimizer_scheduler(model, args)
    normalizer = get_normalizer(multilingual=".en" not in args.model)
    #  -1 is for the special token `sot_prev` and the other half is for the transcribed tokens
    max_prompt_length = model.dims.n_text_ctx // 2 - 1
    fp16 = args.device == "cuda"
    train_data, dev_data = get_dataloaders(
        args, tokenizer, fp16, max_prompt_length, rank
    )
    # to be able to shuffle the data
    train_data.sampler.set_epoch(0)
    train_data = infinite_iter(train_data)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data=train_data,
        dev_data=dev_data,
        normalizer=normalizer,
        tokenizer=tokenizer,
        gpu_id=rank,
        args=args,
    )
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    args = get_parser().parse_args()
    world_size = torch.cuda.device_count()
    print(f"Using {world_size } GPUs")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_args(args, f"{args.save_dir}/args.json")
    mp.spawn(main, args=(args, world_size), nprocs=world_size)
    # writer.close()
