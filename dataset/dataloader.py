import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as HFDataset, load_dataset
from utils.consts import (NUM_MELS, DATASET_TEXT_COL, PAD)
from dataset.char_tokenization import CharacterTokenization
from dataset.conversion_utils import SpeechConverter
from dataset.ds_utils import ds_use
from typing import Callable


def BPE():
    pass


class LibriSpeechDataset(Dataset):
    def __init__(self, hf_ds: HFDataset, text_col: str=DATASET_TEXT_COL, mels: int = NUM_MELS, tokenization_method: Callable = CharacterTokenization().text_to_seq_char_level):
        self.hf_dataset = hf_ds
        self.num_mels = mels
        self.text_col = text_col
        self.tok_fn = tokenization_method
        self.speech_converter = SpeechConverter(self.num_mels)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx) -> tuple[torch.IntTensor, torch.Tensor]:
        #Get the raw text transcription
        text = self.hf_dataset[idx][self.text_col]
        
        #Get the raw audio waveform
        audio_waveform = self.hf_dataset[idx]['audio']['array']
        
        # Apply text_to_seq_fn to the text
        text_seq = self.text_to_seq_fn(text)

        # Processing the wave-form
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=UserWarning)
        #     mel_transform = MelSpectrogram(sampling_rate, n_mels=self.num_mels)
        # mel_spec = mel_transform(audio_waveform)

        #convert raw waveform --> log-mel
        mel_spec = self.speech_converter.convert_to_mel_spec(audio_waveform)
        
        return text_seq, mel_spec

def speech_collate_fn(batch):
    # sort the batch based on input text (this is needed for pack_padded_sequence)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    text_seqs, mel_specs = zip(*batch)
    text_seq_lens = [text_seq.shape[-1] for text_seq in text_seqs] # batch first
    
    #prepare spectrograms for padding
    mel_specs_t = []
    mel_spec_lens = []
    max_mel_seq = -1

    for mel_spec in mel_specs:
        #shape becomes (time, mel_bins)
        mel_specs_t.append(mel_spec.T)

        #number of frames
        true_mel_size = mel_spec.shape[-1]
        mel_spec_lens.append(true_mel_size)
        if true_mel_size > max_mel_seq:
            max_mel_seq = true_mel_size

    #Build stop-token targets
    stop_token_targets = []
    for i in range(len(mel_specs)):
        stop_token_target = torch.zeros(max_mel_seq)
        true_mel_size = mel_spec_lens[i]
        stop_token_target[true_mel_size-1:] = 1
        stop_token_targets.append(stop_token_target)
    
    # pad sequence so pytorch can batch them together
    # alternatives using the minimum from the batch
    # this is using the right padding for samples that have seq_len < max_batch_seq_len  
    tok = CharacterTokenization()
    padded_text_seqs = pad_sequence(text_seqs, batch_first=True, padding_value=tok.symbol_to_idx.get(PAD))
    padded_mel_specs = pad_sequence(mel_specs_t, batch_first=True, padding_value=0)

    #convert lengths to tensors
    text_seq_lens = torch.IntTensor(text_seq_lens)
    mel_spec_lens = torch.IntTensor(mel_spec_lens)
    stop_token_targets = torch.stack(stop_token_targets)

    return padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets

def get_data_loader(dataset: HFDataset, batch_size, shuffle=True, num_workers=0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=speech_collate_fn, 
                      num_workers=num_workers)

def load_data(batch_size, mel_bins=NUM_MELS, subsample_ratio=None):
    #TODO: write this to use the new dataset and convert datasets to dataloader
    train_ds = None
    val_ds = None
    test_ds = None
    train_dl = get_data_loader(train_ds, batch_size, num_workers=3)
    val_dl = get_data_loader(val_ds, batch_size, shuffle=False, num_workers=1)
    test_dl = get_data_loader(test_ds, batch_size, shuffle=False, num_workers=1)
    return train_dl, val_dl, test_dl

