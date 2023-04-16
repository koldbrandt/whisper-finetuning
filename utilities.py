
from dataclasses import asdict
from typing import List


import numpy as np
import jiwer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from whisper.tokenizer import get_tokenizer, Tokenizer
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

from dataloader import get_dataloader



def get_normalizer(multilingual: bool=False):
    if multilingual:
        return BasicTextNormalizer()
    else:
        return EnglishTextNormalizer()

def create_special_token_mask(token_tensor, tokenizer):
    """
    Creates a mask tensor indicating which tokens in the input tensor are special tokens.
    """
    # Create a mask tensor with the same shape as the input tensor
    mask = torch.ones_like(token_tensor, dtype=torch.bool)

    # Smarter hack: use tokenizer.eot
    mask[(token_tensor >= tokenizer.eot) | (token_tensor == -100)] = False
    return mask

def decode_tokens_to_prompt(tokens_batch: List[int], tokenizer: Tokenizer):
    """
    Remove special tokens and decote tokens to text.
    """
    special_token_mask = create_special_token_mask(tokens_batch, tokenizer)
    tokens_without_st = [torch.masked_select(tokens, mask) for tokens, mask in zip(tokens_batch, special_token_mask)]
    text_prompts = [tokenizer.decode(token) for token in tokens_without_st]
    return text_prompts

def get_WER_MultipleTexts(transcription:list, reference:list, normalizer=EnglishTextNormalizer()) -> float: 
    """
    Calculate WER between transcription and reference.
    Transcription and reference are lists of strings.
    """
    if normalizer is not None:
        transcription = [normalizer(text) for text in transcription]
        reference = [normalizer(text) for text in reference]
    wer = jiwer.wer(reference, transcription)
    return wer

def calculate_WER(predicted_tokens, reference_tokens, normalizer, tokenizer) -> float:
    predicted_text_prompts = decode_tokens_to_prompt(predicted_tokens, tokenizer)
    reference_text_prompts = decode_tokens_to_prompt(reference_tokens, tokenizer)
    WER = get_WER_MultipleTexts(predicted_text_prompts, reference_text_prompts, normalizer)
    return WER