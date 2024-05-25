import torch

N = 6
h = 8
epochs = 100
batch_size = 64
d_model = 512
d_ff = 2048
dropout = 0.1
lr = 1.0
max_padding = 72
warmup = 3000
accum_iter = 10
dataset_path = 'multi30k-dataset/data/task1/raw/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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