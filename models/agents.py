from abc import ABC

import torch
import torch.nn as nn
from torch import Tensor

from models.ntm import NTM


class StatefulModule(nn.Module, ABC):
    def init_state(self, batch_size: int) -> None:
        for module in self.children():
            if isinstance(module, StatefulModule):
                module.init_state(batch_size=batch_size)

    def reset_state(self, reset_mask: Tensor) -> None:
        for module in self.children():
            if isinstance(module, StatefulModule):
                module.reset_state(reset_mask=reset_mask)

    def detach_state(self) -> None:
        for module in self.children():
            if isinstance(module, StatefulModule):
                module.detach_state()


class GRUAgent(StatefulModule):
    def __init__(self, input_size: int, state_size: int, output_size: int):
        super().__init__()
        self.initial_state = nn.Parameter(torch.zeros((state_size,)))
        self.gru = nn.GRUCell(input_size, state_size)
        self.char_logits = nn.Linear(state_size, output_size)

    def init_state(self, batch_size: int) -> None:
        super().init_state(batch_size)
        self.state = self.initial_state.expand((batch_size, -1))

    def reset_state(self, reset_mask: Tensor) -> None:
        super().reset_state(reset_mask)
        self.state = torch.where(reset_mask[:, None], self.initial_state, self.state)

    def detach_state(self) -> None:
        super().detach_state()
        self.state = self.state.detach()

    def forward(self, input: Tensor) -> Tensor:
        self.state = self.gru(input, self.state)
        char_logits = self.char_logits(self.state)
        return char_logits


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

    def init_state(self, batch_size: int) -> None:
        super().init_state(batch_size)
        device = self.initial_state.device
        self.initial_memory = torch.zeros((batch_size, self.num_memory_banks, self.memory_bank_size), device=device)
        self.initial_addressing = torch.zeros((batch_size, self.num_memory_banks), device=device)
        self.initial_addressing[:, 0] = 1.0
        self.state = self.initial_state.expand((batch_size, -1))
        self.memory = self.initial_memory
        self.previous_read_addressing = self.initial_addressing
        self.previous_write_addressing = self.initial_addressing

    def reset_state(self, reset_mask: Tensor) -> None:
        super().reset_state(reset_mask)
        self.state = torch.where(reset_mask[:, None], self.initial_state, self.state)
        self.memory = torch.where(reset_mask[:, None, None], self.initial_memory, self.memory)
        self.previous_read_addressing = torch.where(
            reset_mask[:, None], self.initial_addressing, self.previous_read_addressing
        )
        self.previous_write_addressing = torch.where(
            reset_mask[:, None], self.initial_addressing, self.previous_write_addressing
        )

    def detach_state(self) -> None:
        super().detach_state()
        self.state = self.state.detach()
        self.memory = self.memory.detach()
        self.previous_read_addressing = self.previous_read_addressing.detach()
        self.previous_write_addressing = self.previous_write_addressing.detach()

    def forward(self, input: Tensor) -> Tensor:
        output, self.state, self.memory, self.previous_read_addressing, self.previous_write_addressing = self.ntm(
            input=input,
            state=self.state,
            memory=self.memory,
            previous_read_addressing=self.previous_read_addressing,
            previous_write_addressing=self.previous_write_addressing,
        )
        return output
