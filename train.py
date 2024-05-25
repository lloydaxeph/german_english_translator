import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

import configs
from multi30k_custom_dataset import collate_fn, Multi30kDataset
from utils import Batch, LabelSmoothing, SimpleLossCompute, rate
from models import (EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer,
                    MultiHeadedAttention, FeedForwardNetwork, PositionalEncoding,
                    Embeddings, Generator)


def create_datasets() -> (Multi30kDataset, Multi30kDataset, dict, dict):
    train_dataset = Multi30kDataset(os.path.join(configs.dataset_path, 'train.de.gz'),
                                    os.path.join(configs.dataset_path, 'train.en.gz'), max_length=50)
    val_dataset = Multi30kDataset(os.path.join(configs.dataset_path, 'val.de.gz'),
                                  os.path.join(configs.dataset_path, 'val.en.gz'), max_length=50)
    return train_dataset, val_dataset, train_dataset.source_vocab, train_dataset.target_vocab


def create_model(N, h, vocab_src, vocab_tgt) -> (EncoderDecoder, int):
    pad_idx = vocab_tgt["<blank>"]

    attention = MultiHeadedAttention(h=h, d_model=configs.d_model).to(configs.device)
    ffn = FeedForwardNetwork(d_model=configs.d_model, d_ff=configs.d_ff, dropout=configs.dropout).to(configs.device)
    pos_encoding = PositionalEncoding(d_model=configs.d_model, dropout=configs.dropout).to(configs.device)

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


def run_epoch(model: EncoderDecoder, data_iter: Generator, total_batch: int, epoch: int,
              loss_compute: SimpleLossCompute, mode: str, train_state: configs.TrainState, accum_iter: int,
              optimizer: optim.Adam, lr_scheduler: LambdaLR) -> torch.Tensor:
    total_tokens, total_loss, tokens, n_accum = 0, 0, 0, 0
    with tqdm(data_iter, total=total_batch, desc=f'Epoch {epoch + 1}', unit="batch") as tbar:
        for i, batch in enumerate(tbar):
            out = model.forward(src=batch.src.to(configs.device), tgt=batch.tgt.to(configs.device),
                                src_mask=batch.src_mask.to(configs.device),
                                tgt_mask=batch.tgt_mask.to(configs.device))
            loss, loss_node = loss_compute(x=out, y=batch.tgt_y.to(configs.device), norm=batch.n_tokens)
            if mode == 'train':
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.n_tokens
                if i % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                lr_scheduler.step()

            total_loss += loss
            total_tokens += batch.n_tokens
            tokens += batch.n_tokens

            del loss
            del loss_node
        torch.cuda.empty_cache()
    return total_loss / total_tokens, train_state


def validate(model: EncoderDecoder, val_dataset: Multi30kDataset, batch_size: int, pad_idx: int, epoch: int,
             criterion: LabelSmoothing) -> torch.Tensor:
    print('Validating training results...')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model.eval()
    val_loss = run_epoch(model=model, data_iter=(Batch(b[0], b[1], pad_idx) for b in val_dataloader),
                         total_batch=len(val_dataset)//batch_size, epoch=epoch,
                         loss_compute=SimpleLossCompute(model.generator, criterion),
                         mode='eval', train_state=configs.TrainState(), accum_iter=1,
                         optimizer=configs.DummyOptimizer, lr_scheduler=configs.DummyScheduler)
    torch.cuda.empty_cache()
    return val_loss


def run():
    parser = argparse.ArgumentParser(description='Multi-Head Transformer Training.')
    parser.add_argument('--epochs', type=int, default=configs.epochs)
    parser.add_argument('--batch_size', type=int, default=configs.batch_size)
    parser.add_argument('--lr', type=float, default=configs.lr)
    args = parser.parse_args()

    train_dataset, val_dataset, vocab_src, vocab_tgt = create_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model, pad_idx = create_model(N=configs.N, h=configs.h, vocab_src=vocab_src, vocab_tgt=vocab_tgt)
    train_state = configs.TrainState()

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1).to(configs.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step=step, model_size=configs.d_model,
                                                                             factor=1, warmup=configs.warmup))

    for epoch in range(args.epochs):
        model.train()
        train_loss = run_epoch(model=model, data_iter=(Batch(b[0], b[1], pad_idx) for b in train_dataloader),
                               total_batch=len(train_dataset)//args.batch_size, epoch=epoch,
                               loss_compute=SimpleLossCompute(model.generator, criterion), mode='train',
                               train_state=train_state, accum_iter=configs.accum_iter, optimizer=optimizer,
                               lr_scheduler=lr_scheduler)
        file_path = "%s%.2d.pt" % ('multi30k_model_', epoch)
        torch.save(model.state_dict(), file_path)
        torch.cuda.empty_cache()

        val_loss = validate(model=model, val_dataset=val_dataset, batch_size=args.batch_size, pad_idx=pad_idx,
                            epoch=epoch, criterion=criterion)
        print(f'Training loss: {train_loss} | Validation loss: {val_loss}')


if __name__ == '__main__':
    run()
