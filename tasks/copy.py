import numpy as np
import torch
from torch import Tensor

from tasks.base import Task


class CopyTask(Task):
    def __init__(self, vector_size: int, min_sequence_length: int, max_sequence_length: int) -> None:
        super().__init__()
        self.vector_size = vector_size
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        sequence_length = np.random.randint(low=self.min_sequence_length, high=self.max_sequence_length)
        vector_values = np.random.randint(low=1, high=2 ** self.vector_size, size=(sequence_length * batch_size))
        binary_reprs = [np.binary_repr(vector_value, width=self.vector_size) for vector_value in vector_values]
        vectors = np.array([[int(x) for x in vector] for vector in binary_reprs], dtype=bool)
        vectors = vectors.reshape((sequence_length, batch_size, self.vector_size))
        sequence = torch.tensor(vectors, device=device)
        delimiter = torch.zeros((1, batch_size, self.vector_size), dtype=torch.bool, device=device)
        zeros = torch.zeros((1, batch_size, self.vector_size), dtype=torch.bool, device=device)
        inputs = torch.cat([sequence, delimiter, zeros.expand((sequence_length - 1, -1, -1))], dim=0)

        targets = torch.cat([zeros.expand(sequence_length, -1, -1), sequence], dim=0)

        target_mask = torch.zeros((2 * sequence_length, batch_size), dtype=torch.bool, device=device)
        target_mask[sequence_length:] = True

        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.float32)
        target_mask = target_mask.to(torch.float32)

        return inputs, targets, target_mask
