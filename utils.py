from dataclasses import dataclass
import numpy as np
import os


def dist(xpos1, xpos2):
    """Return distances between two xpos arrays.
    Arrays must be same shape of (n,3) or be broadcastable."""
    if len(xpos1.shape) == 2 or len(xpos2.shape) == 2:
        return np.linalg.norm(xpos1 - xpos2, axis=1)
    elif len(xpos1.shape) == 1 and len(xpos2.shape) == 1:
        return np.linalg.norm(xpos1 - xpos2)
    else: raise NotImplementedError

# meta, finetine, multitask
@dataclass(frozen=True)
class TrainConfig:
    num_samples = 5
    meta_batch_size = 2
    n_test_tasks = 2
    n_test_episodes = 2
    epochs = 5_000_000
    do_render = False

    # Evaluation
    num_evals = 10
    num_samples_per_eval = 5

    # Finetune params
    pretrain_epochs = 500_000
    finetune_epochs = 10_000


def make_sequential_log_dir(log_dir):
    """Creates log_dir, appending a number if necessary.

    Attempts to create the directory `log_dir`. If it already exists, appends
    "_1". If that already exists, appends "_2" instead, etc.

    Args:
        log_dir (str): The log directory to attempt to create.

    Returns:
        str: The log directory actually created.

    """
    i = 0
    while True:
        try:
            if i == 0:
                os.makedirs(log_dir)
            else:
                possible_log_dir = '{}_{}'.format(log_dir, i)
                os.makedirs(possible_log_dir)
                log_dir = possible_log_dir
            return log_dir
        except FileExistsError:
            i += 1
