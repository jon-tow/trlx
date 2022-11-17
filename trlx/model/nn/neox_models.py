from typing import *

import torch
from deepspeed.pipe import LayerSpec, PipelineModule  # type: ignore
from megatron import mpu, print_rank_0
from megatron.model.activations import get_activation
from torch import nn


def get_layer_name(layer):
    if isinstance(layer, LayerSpec):
        return layer.typename.__name__
    elif isinstance(layer, nn.Module):
        return layer.__class__.__name__
    else:
        return layer.__name__


class ParallelScalarHead(nn.Module):
    """ScalarHead.

    MLP will take the input with `max_position_embedding` size, project it to
    2*`max_position_embedding`, perform nonlinear transformation, and project
    the state back into `max_position_embedding`. At the end, dropout is also
    applied.
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.bias_gelu_fusion = neox_args.bias_gelu_fusion

        # auto scale so geglu has equal parameters
        ff_mult = 2 * 2 / 3 if self.activation_type == "geglu" else 2
        ff_dim = (
            int(ff_mult * neox_args.max_position_embeddings) * 2
            if self.activation_type == "geglu"
            else ff_mult * neox_args.max_position_embeddings
        )
        self.dense_e_to_2e = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.max_position_embeddings,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        # Project back to `max_position_embeddings`
        self.dense_2e_to_e = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim_in,
            output_size=1,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    def forward(self, hidden_states):
        # [s, b, 2ep]
        intermediate_parallel, bias_parallel = self.dense_e_to_2e(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, e]
        scalar, bias = self.dense_2e_to_e(intermediate_parallel)
        # NOTE: We must return the `hidden_states` to pass on to the
        # final linear layer.
        return scalar, hidden_states, bias


class ParallelScalarHeadPipe(ParallelScalarHead):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "ParallelScalarHead expects a single argument - hidden_states"
        hidden_state = args
        scalar, hidden_states, bias = super().forward(hidden_state)
        return scalar, hidden_states, bias
