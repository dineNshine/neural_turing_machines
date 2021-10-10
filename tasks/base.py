from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Task(ABC):
    @abstractmethod
    def get_batch(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        pass
