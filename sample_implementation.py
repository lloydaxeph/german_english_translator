import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from multi30k_custom_dataset import collate_fn, Multi30kDataset
from multi_head_attention_transformer.utils import Batch, LabelSmoothing, SimpleLossCompute, rate, subsequent_mask
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

        self.attention = MultiHeadedAttention(h=h, d_model=d_model).to(self.device)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout).to(self.device)
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout).to(self.device)

        # Create Model
        deepcopy = copy.deepcopy
        self.model = EncoderDecoder(
            encoder=Encoder(EncoderLayer(size=d_model,
                                         self_attention=deepcopy(self.attention),
                                         feed_forward=deepcopy(self.ffn),
                                         dropout=dropout).to(self.device), N=N).to(self.device),
            decoder=Decoder(DecoderLayer(size=d_model,
                                         self_attention=deepcopy(self.attention),
                                         src_attention=deepcopy(self.attention),
                                         feed_forward=deepcopy(self.ffn),
                                         dropout=dropout).to(self.device), N=N).to(self.device),
            src_embed=nn.Sequential(Embeddings(d_model=d_model, vocab=len(vocab_src)),
                                    deepcopy(self.pos_encoding.to(self.device))).to(self.device),
            tgt_embed=nn.Sequential(Embeddings(d_model=d_model, vocab=len(vocab_tgt)),
                                    deepcopy(self.pos_encoding.to(self.device))).to(self.device),
            generator=Generator(d_model=d_model, vocab=len(vocab_tgt)).to(self.device)
        ).to(self.device)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.criterion = LabelSmoothing(size=len(self.vocab_tgt), padding_idx=self.pad_idx, smoothing=0.1).to(
            self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9
        )

    def train(self, epochs: int, batch_size: int, train_dataset: Multi30kDataset, val_dataset: Multi30kDataset,
              warmup: int = 3000, accum_iter: int = 10):
        module = self.model
        lr_scheduler = LambdaLR(optimizer=self.optimizer,
                                lr_lambda=lambda step: rate(step=step, model_size=self.d_model,
                                                            factor=1, warmup=warmup), )
        print('Training is triggered.')
        train_state = self.__train_loop(module=module, epochs=epochs, train_dataset=train_dataset,
                                        val_dataset=val_dataset, lr_scheduler=lr_scheduler,
                                        accum_iter=accum_iter, batch_size=batch_size)

        file_path = "%sfinal.pt" % 'multi30k_model_'
        torch.save(module.state_dict(), file_path)

    def test(self, test_dataset: Multi30kDataset, vocab_src: any, vocab_tgt: any, n_examples: int = 5,
             model_path: str = None) -> dict:
        test_dataloader = DataLoader(test_dataset, batch_size=n_examples, shuffle=True, collate_fn=collate_fn)
        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )

        print("Checking Model Outputs:")
        example_data = self._check_outputs(
            test_dataloader, vocab_src, vocab_tgt, n_examples=n_examples
        )
        print(example_data)
        return example_data

    def _check_outputs(self, test_dataloader: DataLoader, vocab_src: any, vocab_tgt: any, n_examples: int = 15,
                       pad_idx: int = 2, eos_string: str = '</s>') -> dict:
        results = [()] * n_examples
        for idx in range(n_examples):
            print("\nExample %d ========\n" % idx)
            b = next(iter(test_dataloader))
            rb = Batch(b[0], b[1], pad_idx)
            self.greedy_decode(rb.src, rb.src_mask, 64, 0)[0]

            src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
            tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]

            print(
                "Source Text (Input)        : "
                + " ".join(src_tokens).replace("\n", "")
            )
            print(
                "Target Text (Ground Truth) : "
                + " ".join(tgt_tokens).replace("\n", "")
            )
            model_out = self.greedy_decode(rb.src, rb.src_mask, 72, 0)[0]
            model_txt = (
                    " ".join(
                        [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
                    ).split(eos_string, 1)[0]
                    + eos_string
            )
            print('Model Output               : ' + model_txt.replace('\n', ''))
            results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
        return results

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        memory = self.model.encode(src.to(self.device), src_mask.to(self.device))
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = self.model.decode(memory.to(self.device), src_mask.to(self.device), ys.to(self.device),
                                    subsequent_mask(ys.size(1)).type_as(src.data).to(self.device))
            prob = self.model.generator(out[:, -1].to(self.device))
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys

    def __run_epoch(self, data_iter: Generator, loss_compute: SimpleLossCompute, optimizer: torch.optim,
                    lr_scheduler: LambdaLR, accum_iter: int, train_state: TrainState, mode: str, epoch_state: tuple,
                    total_batch: int = None):

        """Train a single epoch"""
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0
        tqdm_desc = f'Epoch {epoch_state[0] + 1}/{epoch_state[1]}'
        with tqdm(data_iter, total=total_batch, desc=tqdm_desc, unit="batch") as tbar:
            for i, batch in enumerate(tbar):
                out = self.model.forward(src=batch.src.to(self.device), tgt=batch.tgt.to(self.device),
                                         src_mask=batch.src_mask.to(self.device),
                                         tgt_mask=batch.tgt_mask.to(self.device))
                loss, loss_node = loss_compute(x=out, y=batch.tgt_y.to(self.device), norm=batch.n_tokens)
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

    def __run_validation(self, batch_size: int, module: EncoderDecoder, val_dataset: Multi30kDataset, epoch_state: tuple):
        print('Validating training results...')
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.model.eval()
        sloss = self.__run_epoch(
            data_iter=(Batch(b[0], b[1], self.pad_idx) for b in val_dataloader),
            epoch_state=epoch_state,
            loss_compute=SimpleLossCompute(module.generator, self.criterion),
            optimizer=DummyOptimizer(),
            lr_scheduler=DummyScheduler(),
            accum_iter=1,
            train_state=TrainState(),
            mode="eval",
            total_batch=len(val_dataset)//batch_size
        )
        print(f'S-Loss: {sloss}')
        torch.cuda.empty_cache()
        return sloss

    def __train_loop(self, batch_size: int, module: EncoderDecoder, epochs: int, train_dataset: Multi30kDataset,
                     val_dataset: Multi30kDataset, lr_scheduler: LambdaLR, accum_iter: int):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        train_state = TrainState()
        for epoch in range(epochs):
            self.model.train()
            epoch_state = (epoch, epochs)
            _, train_state = self.__run_epoch(
                data_iter=(Batch(b[0], b[1], self.pad_idx) for b in train_dataloader),
                epoch_state=epoch_state,
                loss_compute=SimpleLossCompute(module.generator, self.criterion),
                optimizer=self.optimizer,
                lr_scheduler=lr_scheduler,
                accum_iter=accum_iter,
                train_state=train_state,
                mode='train',
                total_batch=len(train_dataset) // batch_size
            )

            file_path = "%s%.2d.pt" % ('multi30k_model_', epoch)
            torch.save(module.state_dict(), file_path)
            torch.cuda.empty_cache()

            sloss = self.__run_validation(batch_size=batch_size, module=module, val_dataset=val_dataset,
                                          epoch_state=epoch_state)
        return train_state


def create_datasets():
    dataset_path = 'multi30k-dataset/data/task1/raw/'
    train_dataset = Multi30kDataset(os.path.join(dataset_path, 'train.de.gz'),
                                    os.path.join(dataset_path, 'train.en.gz'), max_length=50)
    val_dataset = Multi30kDataset(os.path.join(dataset_path, 'val.de.gz'),
                                  os.path.join(dataset_path, 'val.en.gz'), max_length=50)
    test_dataset = Multi30kDataset(os.path.join(dataset_path, 'test_2018_flickr.de.gz'),
                                   os.path.join(dataset_path, 'test_2018_flickr.en.gz'), max_length=50)
    return train_dataset, val_dataset, test_dataset, train_dataset.source_vocab, train_dataset.target_vocab


if __name__ == '__main__':
    epochs = 10
    batch_size = 32
    max_padding = 72
    train_dataset, val_dataset, test_dataset, vocab_src, vocab_tgt = create_datasets()
    model = MyModel(vocab_src=vocab_src, vocab_tgt=vocab_tgt)
    model.train(epochs=epochs, train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size)
