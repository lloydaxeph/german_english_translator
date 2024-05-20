import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Multi30kDataset(Dataset):
    def __init__(self, source_file, target_file, max_length=None):
        self.source_df = pd.read_csv(source_file, sep='\t', header=None, names=['sentence'])
        self.target_df = pd.read_csv(target_file, sep='\t', header=None, names=['sentence'])
        self.max_length = max_length
        self.source_vocab, self.target_vocab = self.build_vocab()

    def __len__(self):
        return len(self.source_df)

    def build_vocab(self):
        source_counter = Counter()
        target_counter = Counter()

        # Add special tokens to the counters
        special_tokens = ["<s>", "</s>", "<blank>", "<unk>"]
        for token in special_tokens:
            source_counter[token] += 1
            target_counter[token] += 1

        # Count tokens from the dataset
        for idx in range(len(self.source_df)):
            source_sentence = self.source_df.iloc[idx]['sentence']
            target_sentence = self.target_df.iloc[idx]['sentence']
            source_tokens = word_tokenize(source_sentence.lower())
            target_tokens = word_tokenize(target_sentence.lower())

            source_counter.update(source_tokens)
            target_counter.update(target_tokens)

        # Create vocabularies
        source_vocab = {token: idx for idx, (token, _) in enumerate(source_counter.items())}
        target_vocab = {token: idx for idx, (token, _) in enumerate(target_counter.items())}

        return source_vocab, target_vocab

    def __getitem__(self, idx):
        source_sentence = self.source_df.iloc[idx]['sentence']
        target_sentence = self.target_df.iloc[idx]['sentence']
        source_tokens = word_tokenize(source_sentence.lower())
        target_tokens = word_tokenize(target_sentence.lower())

        if self.max_length is not None:
            source_tokens = source_tokens[:self.max_length]
            target_tokens = target_tokens[:self.max_length]

        source_indices = [self.source_vocab.get(token, self.source_vocab["<unk>"]) for token in source_tokens]
        target_indices = [self.target_vocab.get(token, self.target_vocab["<unk>"]) for token in target_tokens]

        return torch.tensor(source_indices), torch.tensor(target_indices)


def collate_fn(batch):
    sources, targets = zip(*batch)
    padded_sources = pad_sequence(sources, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_sources, padded_targets
