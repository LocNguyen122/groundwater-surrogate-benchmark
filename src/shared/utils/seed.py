import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Seed supported RNGs and request deterministic PyTorch execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python inside a PyTorch DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded generator for deterministic DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
