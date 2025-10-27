# -*- coding: utf-8 -*-
"""Concat Convolution 2d Module."""

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

__all__ = ["ConcatConv2d", "ShiftedConv2d", "Conv2dAsLinear"]


class ConcatConv2d(nn.Module):
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: tp.Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert len(in_channels_list) > 1, "ConcatConv2d requires at least 2 input channels"
        self.in_channels_list = in_channels_list
        self.in_channels = sum(in_channels_list)
        num_convs = len(in_channels_list)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias if idx == num_convs - 1 else False,
                    padding_mode,
                    device,
                    dtype,
                )
                for idx, in_channels in enumerate(in_channels_list)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # slice x based on in_channels_list
        x_splits: list[torch.Tensor] = x.split(self.in_channels_list, dim=1)
        # apply each conv to each slice (we have to make contiguous input for quantization)
        out_splits = [conv(x_split.contiguous()) for conv, x_split in zip(self.convs, x_splits, strict=True)]
        # sum the results
        return sum(out_splits)

    @staticmethod
    def from_conv2d(conv: nn.Conv2d, splits: list[int]) -> "ConcatConv2d":
        splits.append(conv.in_channels - sum(splits))
        splits = [s for s in splits if s > 0]
        assert len(splits) > 1, "ConcatConv2d requires at least 2 input channels"
        concat_conv = ConcatConv2d(
            in_channels_list=splits,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        used_in_channels = 0
        for sub_conv in concat_conv.convs:
            assert isinstance(sub_conv, nn.Conv2d)
            in_channels = sub_conv.in_channels
            sub_conv.weight.data.copy_(conv.weight[:, used_in_channels : used_in_channels + in_channels])
            used_in_channels += in_channels
        if conv.bias is not None:
            assert sub_conv.bias is not None
            sub_conv.bias.data.copy_(conv.bias)
        return concat_conv


class ShiftedConv2d(nn.Module):
    shift: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        shift: float | torch.Tensor,
        stride: _size_2_t = 1,
        padding: tp.Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.conv.shifted = True
        if not isinstance(shift, torch.Tensor):
            shift = torch.tensor(shift, device=device, dtype=dtype)
        shift = shift.flatten().to(device=device, dtype=dtype)
        shift_channels = shift.numel()
        if shift_channels > 1:
            assert padding == 0, "Padding is not supported for multi-channel shift"
            assert in_channels >= shift_channels and in_channels % shift_channels == 0
            shift = shift.view(shift_channels, 1).expand(shift_channels, in_channels // shift_channels)
            shift = shift.reshape(1, in_channels, 1, 1)
        self.register_buffer("shift", shift.view(1, -1, 1, 1))
        # region update padding-related attributes
        self.padding_size = self.conv._reversed_padding_repeated_twice
        self.padding_mode, self.padding_value = self.conv.padding_mode, None
        if all(p == 0 for p in self.padding_size):
            self.padding_mode = ""
        elif self.padding_mode == "zeros":
            self.padding_mode = "constant"
            assert shift.numel() == 1, "Zero padding is not supported for multi-channel shift"
            self.padding_value = shift.item()
        self.conv.padding = "valid"
        self.conv.padding_mode = "zeros"
        self.conv._reversed_padding_repeated_twice = [0, 0] * len(self.conv.kernel_size)
        # endregion

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input + self.shift
        if self.padding_mode:
            input = F.pad(input, self.padding_size, mode=self.padding_mode, value=self.padding_value)
        return self.conv(input)

    @staticmethod
    def from_conv2d(conv: nn.Conv2d, shift: float | torch.Tensor) -> "ShiftedConv2d":
        device, dtype = conv.weight.device, conv.weight.dtype
        shifted = ShiftedConv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            shift=shift,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
            device=device,
            dtype=dtype,
        )
        shifted.conv.weight.data.copy_(conv.weight)
        shift = shifted.shift
        if shift.numel() == 1:
            shifted_bias = conv.weight.double().sum(dim=[1, 2, 3]) * shift.double()
        else:
            shifted_bias = torch.matmul(conv.weight.double().sum(dim=[2, 3]), shift.view(-1).double())
        shifted_bias = shifted_bias.view(shifted.conv.bias.size())
        if conv.bias is not None:
            shifted.conv.bias.data.copy_((conv.bias.data.double() - shifted_bias).to(dtype))
        else:
            shifted.conv.bias.data.copy_(-shifted_bias.to(dtype))
        return shifted


class Conv2dAsLinear(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        assert isinstance(conv, nn.Conv2d)
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.bias_flag = conv.bias is not None

        # -------------------------------
        # unfold Conv2d's weight and bias to Linear
        # Conv2d.weight.shape = [out_channels, in_channels, kH, kW]
        # Linear.weight.shape = [out_features, in_features]
        # in_features = in_channels * kH * kW
        # out_features = out_channels
        # -------------------------------
        kH, kW = self.kernel_size
        self.linear = nn.Linear(
            in_features=self.in_channels * kH * kW,
            out_features=self.out_channels,
            bias=self.bias_flag,
            device=conv.weight.device,
            dtype=conv.weight.dtype
        )

        # copy parameters
        with torch.no_grad():
            self.linear.weight.copy_(conv.weight.view(self.out_channels, -1))
            if self.bias_flag:
                self.linear.bias.copy_(conv.bias)

    def forward(self, x: torch.Tensor):
        # input: x.shape = [N, C_in, H, W]
        # output: y.shape = [N, C_out, H_out, W_out]
        N, C, H, W = x.shape

        # 1. unfold
        # after unfolding: x_unf.shape = [N, C*kH*kW, L]，其中 L = H_out * W_out
        x_unf = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # 2. transpose: [N, L, C*kH*kW]
        x_unf = x_unf.transpose(1, 2)

        # 3. pass through linear layer
        # output shape: [N, L, C_out]
        y: torch.Tensor = self.linear(x_unf)

        # 4. transpose back and reshape
        # output shape: [N, C_out, H_out, W_out]
        H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        y = y.transpose(1, 2).reshape(N, self.out_channels, H_out, W_out)

        return y