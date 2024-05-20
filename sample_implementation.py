import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from multi30k_custom_dataset import collate_fn, Multi30kDataset
from multi_head_attention_transformer.utils import Batch, LabelSmoothing, SimpleLossCompute, rate
from multi_head_attention_transformer.models import (EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer,
                                                 MultiHeadedAttention, FeedForwardNetwork, PositionalEncoding,
                                                 Embeddings, Generator)

"""
This project aims to train and use multi-head attention transformer model to translate German text to English.
Dataset that will be used in this implemenation is the Multi30k dataset.
"""


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class TrainState:
    """Track number of steps, examples, and tokens processed"""
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class MyModel:
    def __init__(self, vocab_src, vocab_tgt, N: int = 6, d_model: int = 512, d_ff: int = 2048, h: int = 8,
                 dropout: float = 0.1, base_lr: float = 1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.pad_idx = self.vocab_tgt["<blank>"]

        self.attention = MultiHeadedAttention(h=h, d_model=d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Create Model
        deepcopy = copy.deepcopy
        self.model = EncoderDecoder(
            encoder=Encoder(EncoderLayer(size=d_model,
                                         self_attention=deepcopy(self.attention),
                                         feed_forward=deepcopy(self.ffn),
                                         dropout=dropout), N=N),
            decoder=Decoder(DecoderLayer(size=d_model,
                                         self_attention=deepcopy(self.attention),
                                         src_attention=deepcopy(self.attention),
                                         feed_forward=deepcopy(self.ffn),
                                         dropout=dropout), N=N),
            src_embed=nn.Sequential(Embeddings(d_model=d_model, vocab=len(vocab_src)), deepcopy(self.pos_encoding)),
            tgt_embed=nn.Sequential(Embeddings(d_model=d_model, vocab=len(vocab_tgt)), deepcopy(self.pos_encoding)),
            generator=Generator(d_model=d_model, vocab=len(vocab_tgt))
        )
        self.model = self.model.to(self.device)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.criterion = LabelSmoothing(size=len(self.vocab_tgt), padding_idx=self.pad_idx, smoothing=0.1).to(
            self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9
        )

    def train(self, epochs: int, train_dataloader: DataLoader, valid_dataloader: DataLoader,
              warmup: int = 3000, accum_iter: int = 10):
        module = self.model
        lr_scheduler = LambdaLR(optimizer=self.optimizer,
                                lr_lambda=lambda step: rate(step=step, model_size=self.d_model,
                                                            factor=1, warmup=warmup), )
        print('Training is triggered.')
        train_state = self.__train_loop(module=module, epochs=epochs, train_dataloader=train_dataloader,
                                        valid_dataloader=valid_dataloader, lr_scheduler=lr_scheduler,
                                        accum_iter=accum_iter)

        file_path = "%sfinal.pt" % 'multi30k_model_'
        torch.save(module.state_dict(), file_path)

    def __run_epoch(self, data_iter: Generator, loss_compute: SimpleLossCompute, optimizer: torch.optim,
                    lr_scheduler: LambdaLR,
                    accum_iter: int, train_state: TrainState, mode: str, epoch_state: tuple):

        """Train a single epoch"""
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0
        tqdm_desc = f'Epoch {epoch_state[0] + 1}/{epoch_state[1]}'
        with tqdm(data_iter, total=total_batch, desc=tqdm_desc, unit="batch") as tbar:
            for i, batch in enumerate(tbar):
                out = self.model.forward(src=batch.src, tgt=batch.tgt, src_mask=batch.src_mask, tgt_mask=batch.tgt_mask)
                loss, loss_node = loss_compute(x=out, y=batch.tgt_y, norm=batch.n_tokens)
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
        return total_loss / total_tokens, train_state

    def __run_validation(self, module: EncoderDecoder, valid_dataloader: DataLoader, epoch_state: tuple):
        print('Validating training results...')
        self.model.eval()
        sloss = self.__run_epoch(
            data_iter=(Batch(b[0], b[1], self.pad_idx) for b in valid_dataloader),
            epoch_state=epoch_state,
            loss_compute=SimpleLossCompute(module.generator, self.criterion),
            optimizer=DummyOptimizer(),
            lr_scheduler=DummyScheduler(),
            accum_iter=1,
            train_state=TrainState(),
            mode="eval"
        )
        print(f'S-Loss: {sloss}')
        torch.cuda.empty_cache()
        return sloss

    def __train_loop(self, module: EncoderDecoder, epochs: int, train_dataloader: DataLoader,
                     valid_dataloader: DataLoader, lr_scheduler: LambdaLR, accum_iter: int):
        self.model.train()
        train_state = TrainState()
        for epoch in range(epochs):
            epoch_state = (epoch, epochs)
            _, train_state = self.__run_epoch(
                data_iter=(Batch(b[0], b[1], self.pad_idx) for b in train_dataloader),
                epoch_state=epoch_state,
                loss_compute=SimpleLossCompute(module.generator, self.criterion),
                optimizer=self.optimizer,
                lr_scheduler=lr_scheduler,
                accum_iter=accum_iter,
                train_state=train_state,
                mode='train'
            )

            file_path = "%s%.2d.pt" % ('multi30k_model_', epoch)
            torch.save(module.state_dict(), file_path)
            torch.cuda.empty_cache()

            sloss = self.__run_validation(module=module, valid_dataloader=valid_dataloader, epoch_state=epoch_state)
        return train_state


if __name__ == '__main__':
    epochs = 10
    batch_size = 64
    max_padding = 72

    train_dataset = Multi30kDataset("multi30k-dataset/data/task1/raw/train.de.gz",
                                    "multi30k-dataset/data/task1/raw/train.en.gz", max_length=50)
    total_batch = len(train_dataset) // batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = Multi30kDataset("multi30k-dataset/data/task1/raw/val.de.gz",
                                  "multi30k-dataset/data/task1/raw/val.en.gz", max_length=50)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    vocab_src, vocab_tgt = train_dataset.source_vocab, train_dataset.target_vocab
    model = MyModel(vocab_src=vocab_src, vocab_tgt=vocab_tgt)
    model.train(epochs=epochs, train_dataloader=train_loader, valid_dataloader=val_loader)
