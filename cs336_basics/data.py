import os
import time
from typing import IO, BinaryIO
import torch
import numpy as np
import numpy.typing as npt


def bytes_to_tuple(bytes: bytes):
    return tuple(map(int.to_bytes, bytes))


def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([
            torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
            for i in starting_idxs
    ])  # fmt: skip
    y = torch.stack(
        [
            torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
            for i in starting_idxs
        ]
    )  # fmt: skip
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


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


def combine_npy_files(npy_files, output_path):
    print(f"Combining {len(npy_files)} chunks into final output file {output_path}.")
    start_time = time.perf_counter()

    # 1. Compute total size for preallocation
    total_len = sum(np.load(f, mmap_mode="r").shape[0] for f in npy_files)

    # 2. Create temporary memmap file (raw binary buffer)
    tmp_path = output_path + ".tmp"
    final = np.memmap(tmp_path, dtype=np.uint16, mode="w+", shape=(total_len,))

    # 3. Copy chunks one by one
    offset = 0
    for path in npy_files:
        arr: np.ndarray = np.load(path, mmap_mode="r")
        final[offset:offset + len(arr)] = arr.astype(np.uint16, copy=False)
        offset += len(arr)

    # 4. Flush changes to disk
    final.flush()
    del final  # close memmap

    # 5. Reload the memmap file and save as proper .npy
    clean_array = np.memmap(tmp_path, dtype=np.uint16, mode="r", shape=(total_len,))
    np.save(output_path, np.asarray(clean_array))  # save as real .npy
    del clean_array
    os.remove(tmp_path)  # cleanup

    end_time = time.perf_counter()
    print(f"Time taken for combining files: {end_time - start_time:.2f} seconds")
