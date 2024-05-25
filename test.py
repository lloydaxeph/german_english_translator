import os
import copy
import argparse
import torch
import torch.nn as nn
import torchtext
from torch.utils.data import DataLoader

import configs
from multi30k_custom_dataset import collate_fn, Multi30kDataset
from utils import Batch, subsequent_mask
from models import (EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer,
                    MultiHeadedAttention, FeedForwardNetwork, PositionalEncoding,
                    Embeddings, Generator)


def create_datasets(data_path: str) -> (Multi30kDataset, dict, dict):
    train_dataset = Multi30kDataset(os.path.join(configs.dataset_path, 'train.de.gz'),
                                    os.path.join(configs.dataset_path, 'train.en.gz'), max_length=50)
    test_dataset = Multi30kDataset(os.path.join(data_path, 'test_2018_flickr.de.gz'),
                                   os.path.join(data_path, 'test_2018_flickr.en.gz'), max_length=50)
    return test_dataset, train_dataset.source_vocab, train_dataset.target_vocab, \
           test_dataset.source_vocab, test_dataset.target_vocab


def create_model(N, h, vocab_src, vocab_tgt) -> (EncoderDecoder, int):
    pad_idx = vocab_tgt["<blank>"]

    attention = MultiHeadedAttention(h=h, d_model=configs.d_model).to(configs.device)
    ffn = FeedForwardNetwork(d_model=configs.d_model, d_ff=configs.d_ff, dropout=configs.dropout).to(configs.device)
    pos_encoding = PositionalEncoding(d_model=configs.d_model, dropout=configs.dropout).to(configs.device)

    # Create Model
    deepcopy = copy.deepcopy
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(size=configs.d_model,
                                     self_attention=deepcopy(attention),
                                     feed_forward=deepcopy(ffn),
                                     dropout=configs.dropout).to(configs.device), N=N).to(configs.device),
        decoder=Decoder(DecoderLayer(size=configs.d_model,
                                     self_attention=deepcopy(attention),
                                     src_attention=deepcopy(attention),
                                     feed_forward=deepcopy(ffn),
                                     dropout=configs.dropout).to(configs.device), N=N).to(configs.device),
        src_embed=nn.Sequential(Embeddings(d_model=configs.d_model, vocab=len(vocab_src)),
                                deepcopy(pos_encoding.to(configs.device))).to(configs.device),
        tgt_embed=nn.Sequential(Embeddings(d_model=configs.d_model, vocab=len(vocab_tgt)),
                                deepcopy(pos_encoding.to(configs.device))).to(configs.device),
        generator=Generator(d_model=configs.d_model, vocab=len(vocab_tgt)).to(configs.device)
    ).to(configs.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model, pad_idx


def greedy_decode(model: EncoderDecoder, src: torch.Tensor, src_mask: torch.Tensor, max_len: int,
                  start_symbol: int) -> torch.Tensor:
    memory = model.encode(src.to(configs.device), src_mask.to(configs.device))
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data).to(configs.device)
    for i in range(max_len - 1):
        out = model.decode(memory.to(configs.device), src_mask.to(configs.device), ys.to(configs.device),
                           subsequent_mask(ys.size(1)).type_as(src.data).to(configs.device))
        prob = model.generator(out[:, -1].to(configs.device))
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(0).to(configs.device)], dim=1)
    return ys


def check_outputs(model: EncoderDecoder, test_dataloader: DataLoader, num_samples: int, vocab_src: dict,
                  vocab_tgt: dict, eos_string: str = '</s>', pad_idx: int = 2) -> list:
    results = [()] * num_samples
    vocab_src = {v: k for k, v in vocab_src.items()}
    vocab_tgt = {v: k for k, v in vocab_tgt.items()}
    for idx in range(num_samples):
        b = next(iter(test_dataloader))
        rb = Batch(b[0], b[1], pad_idx)

        src_tokens = [vocab_src[int(x)] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt[int(x)] for x in rb.tgt[0] if x != pad_idx]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )

        model_out = greedy_decode(model=model, src=rb.src, src_mask=rb.src_mask, max_len=72, start_symbol=0)[0]
        model_txt = (
                " ".join([vocab_src[int(x)] for x in model_out if x != pad_idx]).split(eos_string, 1)[0]
                + eos_string
        )
        print('Model Output               : ' + model_txt.replace('\n', ''))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_test():
    parser = argparse.ArgumentParser(description='Multi-Head Transformer Training Test.')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=configs.dataset_path)
    parser.add_argument('--num_samples', type=int, default=configs.batch_size)
    parser.add_argument('--print', type=bool, default=False)
    args = parser.parse_args()

    test_dataset, vocab_src, vocab_tgt, test_vocab_src, test_vocab_tgt = create_datasets(data_path=args.data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model, pad_idx = create_model(N=configs.N, h=configs.h, vocab_src=vocab_src, vocab_tgt=vocab_tgt)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    results = check_outputs(model=model, test_dataloader=test_dataloader, num_samples=args.num_samples,
                            vocab_src=test_vocab_src, vocab_tgt=test_vocab_tgt)
    if args.print:
        print(f'RESULTS: {results}')


if __name__ == '__main__':
    run_test()
