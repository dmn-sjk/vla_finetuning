"""Samples/s over one or more timed segments, synchronized across ranks at each edge."""

import time
from collections.abc import Callable

import torch

from src import dist_utils as D


class ThroughputTimer:
    def __init__(self, device: torch.device, barrier: Callable[[], None] = D.barrier):
        self.device = device
        self.barrier = barrier
        self.started = False
        self.elapsed = 0.0
        self.samples = 0
        self._seg_start = None

    @property
    def running(self) -> bool:
        return self._seg_start is not None

    @property
    def samples_per_s(self) -> float:
        return self.samples / max(self.elapsed, 1e-9)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.barrier()

    def start(self) -> None:
        self._sync()
        self.started, self._seg_start, self.samples = True, time.time(), 0

    def pause(self) -> None:
        if self._seg_start is None:
            return
        self._sync()
        self.elapsed += time.time() - self._seg_start
        self._seg_start = None

    def resume(self) -> None:
        if self.started and self._seg_start is None:
            self._seg_start = time.time()

    def add(self, n_samples: int) -> None:
        if self.started:
            self.samples += n_samples
