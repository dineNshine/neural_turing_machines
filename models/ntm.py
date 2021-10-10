import math

import torch
import torch.nn as nn
from torch import Tensor


class AddressingHead(nn.Module):
    def __init__(self, input_size: int, max_shift: int, memory_bank_size: int):
        super().__init__()
        num_shift_offsets = 2 * max_shift + 1
        self.memory_bank_size = memory_bank_size
        self.argument_sizes = [memory_bank_size, 1, num_shift_offsets]
        self.arguments = nn.Linear(input_size, sum(self.argument_sizes))
        nn.init.constant_(self.arguments.bias[memory_bank_size], -0.5 * math.log(self.memory_bank_size))

    def forward(
        self,
        input: Tensor,  # (batch_size, input_size)
        memory: Tensor,  # (batch_size, num_memory_banks, memory_bank_size)
        previous_addressing: Tensor,  # (batch_size, num_memory_banks)
    ) -> Tensor:
        arguments = self.arguments(input).split(self.argument_sizes, dim=1)
        # (batch_size, memory_bank_size), (batch_size, 1), (batch_size, num_shift_offsets)
        query, query_gate, shift_offset_distribution = arguments
        query_gate = query_gate.sigmoid()
        shift_offset_distribution = shift_offset_distribution.softmax(dim=1)

        memory_bank_size = memory.shape[2]
        # (batch_size, num_memory_banks)
        content_based_addr = ((memory @ query[:, :, None]).squeeze(-1) / math.sqrt(memory_bank_size)).softmax(dim=1)
        # (batch_size, num_memory_banks)
        interpolated_addr = query_gate * content_based_addr + (1 - query_gate) * previous_addressing

        num_memory_banks = previous_addressing.shape[1]
        num_shift_offsets = shift_offset_distribution.shape[1]
        max_shift = num_shift_offsets // 2

        # (num_memory_banks,)
        memory_bank_idxs = torch.arange(num_memory_banks)
        # (num_shift_offsets,)
        shift_offsets = torch.arange(num_shift_offsets) - max_shift

        # (num_memory_banks, num_shift_offsets)
        memory_bank_idx_table = (memory_bank_idxs[:, None] + shift_offsets[None, :]) % num_memory_banks
        # (bs, num_memory_banks)
        shifted_addr = (shift_offset_distribution[:, None, :] * interpolated_addr[:, memory_bank_idx_table]).sum(dim=2)

        return shifted_addr


class ReadHead(nn.Module):
    def __init__(self, input_size: int, max_shift: int, memory_bank_size: int):
        super().__init__()
        self.addressing_head = AddressingHead(input_size, max_shift, memory_bank_size)

    def forward(
        self,
        input: Tensor,  # (batch_size, input_size)
        memory: Tensor,  # (batch_size, num_memory_banks, memory_bank_size)
        previous_addressing: Tensor,  # (batch_size, num_memory_banks)
    ) -> tuple[Tensor, Tensor]:
        # (bs, num_memory_banks)
        addressing = self.addressing_head(input, memory, previous_addressing)
        # (bs, memory_bank_size)
        data = (addressing[:, None] @ memory).squeeze(1)
        return data, addressing


class WriteHead(nn.Module):
    def __init__(self, input_size: int, max_shift: int, memory_bank_size: int):
        super().__init__()
        self.memory_bank_size = memory_bank_size
        self.addressing_head = AddressingHead(input_size, max_shift, memory_bank_size)
        self.ea = nn.Linear(input_size, 2 * memory_bank_size)

    def forward(
        self,
        input: Tensor,  # (batch_size, input_size)
        memory: Tensor,  # (batch_size, num_memory_banks, memory_bank_size)
        previous_addressing: Tensor,  # (batch_size, num_memory_banks)
    ) -> tuple[Tensor, Tensor]:
        # (bs, num_memory_banks)
        addressing = self.addressing_head(input, memory, previous_addressing)
        # (bs, 2 * memory_bank_size)
        ea = self.ea(input)
        # (bs, memory_bank_size)
        erase = ea[:, : self.memory_bank_size].sigmoid()
        add = ea[:, self.memory_bank_size :]
        # (bs, num_memory_banks, memory_bank_size)
        memory = (1 - addressing[:, :, None] * erase[:, None, :]) * memory + addressing[:, :, None] * add[:, None, :]
        return memory, addressing


class NTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        memory_bank_size: int,
        max_shift: int,
        output_size: int,
    ):
        super().__init__()
        self.read_head_features = nn.Linear(input_size + state_size, state_size)
        self.read_head = ReadHead(state_size, max_shift, memory_bank_size)
        self.gru = nn.GRUCell(input_size + memory_bank_size, state_size)
        self.write_head = WriteHead(state_size, max_shift, memory_bank_size)
        self.output = nn.Linear(state_size, output_size)

    def forward(
        self,
        input: Tensor,  # (batch_size, input_size)
        state: Tensor,  # (batch_size, state_size)
        memory: Tensor,  # (batch_size, num_memory_banks, memory_bank_size)
        previous_read_addressing: Tensor,  # (batch_size, num_memory_banks)
        previous_write_addressing: Tensor,  # (batch_size, num_memory_banks)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # (batch_size, input_size + state_size)
        input_state0 = torch.cat([input, state], dim=1)
        # (batch_size, state_size)
        read_head_features = torch.relu(self.read_head_features(input_state0))
        # (batch_size, memory_bank_size), (batch_size, num_memory_banks)
        read_data, read_addressing = self.read_head(read_head_features, memory, previous_read_addressing)
        # (batch_size, input_size + memory_bank_size)
        input_read_data = torch.cat([input, read_data], dim=1)
        # (batch_size, state_size)
        state = self.gru(input_read_data, state)
        # (batch_size, num_memory_banks, memory_bank_size), (batch_size, num_memory_banks)
        memory, write_addressing = self.write_head(state, memory, previous_write_addressing)
        # (batch_size, output_size)
        output = self.output(state)
        return output, state, memory, read_addressing, write_addressing
