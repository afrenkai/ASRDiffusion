from typing import Callable, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import (
    Lowercase,
    Sequence as NormalizerSequence,
    NFD,
    StripAccents,
)
import argparse
import os
from utils.consts import DATADIR, PAD, EOS, DATASET_ROOT


def build_tokenizer_vocab(
    type: Callable,
    unk_tok: str,
    special_toks: list[str],
    vocab_size: int,
    min_freq: int,
    files: Optional[str | list[str]] = None,
    save_dir: Optional[str] = None,
):
    """
    rtfm,
    https://huggingface.co/docs/tokenizers/en/quicktour

    but tldr;
    trains a BPE tokenizer

    saves vocab to a json

    can be called with

    tokenizer = Tokenizer.from_file("<PATH TO JSON>")
    from there, can be accessed aud used to encode and decode content, i.e.

    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    """

    tok = Tokenizer(type(unk_token=unk_tok))
    trainer = BpeTrainer(
        special_tokens=special_toks, vocab_size=vocab_size, min_frequency=min_freq
    )
    tok.pre_tokenizer = Whitespace()

    tok.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])

    if files:
        tok.train(files, trainer)
    # print(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fpath = os.path.join(save_dir, "tokenizer.json")
        tok.save(fpath)

    return tok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for LibriSpeech or custom text data"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="Path to text files to use for tokenizing",
        default=None,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="HuggingFace dataset name (e.g., 'openslr/librispeech_asr')",
        default=DATASET_ROOT,
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        help="Dataset split to use for training (e.g., 'train.clean.100')",
        default="train.clean.100",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        help="Name of text column in dataset",
        default="text",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Max size of the vocab for the tokenizer to use",
        default=5000,
        # required=True,
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        help="Minimum frequency for tokens to be included",
        default=2,
    )
    parser.add_argument(
        "--unk_token",
        type=str,
        help="Unknown token symbol",
        default="[UNK]",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to save the tokenizer JSON file",
        default=DATADIR,
    )

    args = parser.parse_args()

    special_tokens = [PAD, EOS, args.unk_token, "[CLS]", "[SEP]"]

    print(
        f"Training BPE tokenizer with vocab_size={args.vocab_size}, min_freq={args.min_freq}"
    )

    if not args.files:
        raise ValueError("pass in local files to use for tokenization")
    if args.save_dir:
        save_location = args.save_dir
    else:
        save_location = DATADIR

    tokenizer = build_tokenizer_vocab(
        type=BPE,
        unk_tok=args.unk_token,
        special_toks=special_tokens,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        files=args.files,
        save_dir=save_location,
    )

    print(f"Tokenizer saved to: {save_location}")

    test_text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    output = tokenizer.encode(test_text)
    print(f"\nTest encoding: '{test_text}'")
    print(f"Tokens: {output.tokens}")
    print(f"IDs: {output.ids}")
