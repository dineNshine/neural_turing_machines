from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from models.ntm import NTM


class StatefulModule(nn.Module, ABC):
    @abstractmethod
    def init_state(self, batch_size: int) -> Any:
        pass

    @abstractmethod
    def reset_state(self, state: Any, reset_mask: Tensor) -> Any:
        pass

    @abstractmethod
    def detach_state(self, state: Any) -> Any:
        pass


class GRUAgent(StatefulModule):
    def __init__(self, input_size: int, state_size: int, output_size: int):
        super().__init__()
        self.initial_state = nn.Parameter(torch.zeros((state_size,)))
        self.gru = nn.GRUCell(input_size, state_size)
        self.char_logits = nn.Linear(state_size, output_size)

    def init_state(self, batch_size: int) -> Tensor:
        return self.initial_state.expand((batch_size, -1))

    def reset_state(self, state: Tensor, reset_mask: Tensor) -> Tensor:
        return torch.where(reset_mask[:, None], self.initial_state, state)

    def detach_state(self, state: Tensor) -> Tensor:
        return state.detach()

    def forward(self, input: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        state = self.gru(input, state)
        char_logits = self.char_logits(state)
        return char_logits, state


class NTMAgent(StatefulModule):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        num_memory_banks: int,
        memory_bank_size: int,
        max_shift: int,
        output_size: int,
    ):
        super().__init__()
        self.num_memory_banks = num_memory_banks
        self.memory_bank_size = memory_bank_size

        self.initial_state = nn.Parameter(torch.zeros((state_size,)))
        self.ntm = NTM(
            input_size=input_size,
            output_size=output_size,
            state_size=state_size,
            memory_bank_size=memory_bank_size,
            max_shift=max_shift,
        )

    def init_state(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = self.initial_state.device
        self.initial_memory = torch.zeros((batch_size, self.num_memory_banks, self.memory_bank_size), device=device)
        self.initial_addressing = torch.zeros((batch_size, self.num_memory_banks), device=device)
        self.initial_addressing[:, 0] = 1.0
        state_ = self.initial_state.expand((batch_size, -1))
        memory = self.initial_memory
        previous_read_addressing = self.initial_addressing
        previous_write_addressing = self.initial_addressing
        return state_, memory, previous_read_addressing, previous_write_addressing

    def reset_state(
        self, state: tuple[Tensor, Tensor, Tensor, Tensor], reset_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state_, memory, previous_read_addressing, previous_write_addressing = state
        state_ = torch.where(reset_mask[:, None], self.initial_state, state_)
        memory = torch.where(reset_mask[:, None, None], self.initial_memory, memory)
        previous_read_addressing = torch.where(reset_mask[:, None], self.initial_addressing, previous_read_addressing)
        previous_write_addressing = torch.where(reset_mask[:, None], self.initial_addressing, previous_write_addressing)
        return state_, memory, previous_read_addressing, previous_write_addressing

    def detach_state(self, state: tuple[Tensor, Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state_, memory, previous_read_addressing, previous_write_addressing = state
        state_ = state_.detach()
        memory = memory.detach()
        previous_read_addressing = previous_read_addressing.detach()
        previous_write_addressing = previous_write_addressing.detach()
        return state_, memory, previous_read_addressing, previous_write_addressing

    def forward(
        self, input: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        state_, memory, previous_read_addressing, previous_write_addressing = state
        output, state_, memory, previous_read_addressing, previous_write_addressing = self.ntm(
            input=input,
            state=state_,
            memory=memory,
            previous_read_addressing=previous_read_addressing,
            previous_write_addressing=previous_write_addressing,
        )
        return output, (state_, memory, previous_read_addressing, previous_write_addressing)
