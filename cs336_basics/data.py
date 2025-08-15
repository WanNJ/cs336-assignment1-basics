import os
from typing import IO, BinaryIO
import torch
import numpy as np
import numpy.typing as npt


def bytes_to_tuple(bytes: bytes):
    return tuple(map(int.to_bytes, bytes))


def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(dataset) < context_length + batch_size:
        raise ValueError(f"Not possible to generate {batch_size} unique samples.") 

    valid_indices = np.arange(len(dataset) - context_length)
    sampled_start_indices = np.random.choice(valid_indices, batch_size, replace=False)
    x = np.array([dataset[s:s+context_length] for s in sampled_start_indices])
    y = np.array([dataset[s+1:s+context_length+1] for s in sampled_start_indices])
    return torch.tensor(x, device=device), torch.tensor(y, device=device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    states = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(states, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    saved_states = torch.load(src)
    model.load_state_dict(saved_states["model"])
    optimizer.load_state_dict(saved_states["optimizer"])
    return saved_states["iteration"]
