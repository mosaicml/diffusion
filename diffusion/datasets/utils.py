# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Useful functions for dealing with streaming datasets."""

from pathlib import Path
from typing import Sequence

from streaming import Stream


def make_streams(remote, local=None, proportion=None, repeat=None, choose=None):
    """Helper function to create a list of Stream objects from a set of remotes and stream weights.

    Args:
        remote (Union[str, Sequence[str]]): The remote path or paths to stream from.
        local (Union[str, Sequence[str]], optional): The local path or paths to cache the data. If not provided, the
            default local path is used. Default: ``None``.
        proportion (list, optional): Specifies how to sample this Stream relative to other Streams. Default: ``None``.
        repeat (list, optional): Specifies the degree to which a Stream is upsampled or downsampled. Default: ``None``.
        choose (list, optional): Specifies the number of samples to choose from a Stream. Default: ``None``.

    Returns:
        List[Stream]: A list of Stream objects.
    """
    remote, local = _make_remote_and_local_sequences(remote, local)
    proportion, repeat, choose = _make_weighting_sequences(remote, proportion, repeat, choose)

    streams = []
    for i, (r, l) in enumerate(zip(remote, local)):
        streams.append(Stream(remote=r, local=l, proportion=proportion[i], repeat=repeat[i], choose=choose[i]))
    return streams


def _make_remote_and_local_sequences(remote, local=None):
    if isinstance(remote, str):
        remote = [remote]
    if isinstance(local, str):
        local = [local]
    if not local:
        local = [_make_default_local_path(r) for r in remote]

    if isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            ValueError(
                f'remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}')
    else:
        ValueError(f'remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.')
    return remote, local


def _make_default_local_path(remote_path):
    return str(Path(*['/tmp'] + list(Path(remote_path).parts[1:])))


def _make_weighting_sequences(remote, proportion=None, repeat=None, choose=None):
    weights = {'proportion': proportion, 'repeat': repeat, 'choose': choose}
    for name, weight in weights.items():
        if weight is not None and len(remote) != len(weight):
            ValueError(f'{name} must be the same length as remote, got lengths {len(remote)} and {len(weight)}')
    proportion = weights['proportion'] if weights['proportion'] is not None else [None] * len(remote)
    repeat = weights['repeat'] if weights['repeat'] is not None else [None] * len(remote)
    choose = weights['choose'] if weights['choose'] is not None else [None] * len(remote)
    return proportion, repeat, choose
