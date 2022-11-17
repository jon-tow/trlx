"""
Usage:
    python gpt-neox/deepy.py tests/test_neox.py -d configs neox_ppo_model_config.yml
"""
import unittest

import trlx
from trlx.data.configs import TRLConfig
from trlx.utils.megatron import enable_neox
from trlx.utils.loading import get_model, get_orchestrator
from megatron.text_generation_utils import get_batch
from megatron.utils import print_rank_0

enable_neox("gpt-neox")

import torch
from megatron.neox_arguments import NeoXArgs
from transformers import AutoTokenizer


def test_specs(neox_args, config):
    model_class = get_model(config.model.model_type)
    rl_model = model_class(config, neox_args)
    text = ["Hello, my dog", "How are you doing today?"]
    tokens = [rl_model.tokenizer(t) for t in text]
    tokens = rl_model.tokenizer.pad(tokens)['input_ids']
    tokens, attention_mask, position_ids = get_batch(
        rl_model.neox_args, tokens)
    outputs = rl_model.model((tokens, position_ids, attention_mask))
    breakpoint()


if __name__ == "__main__":
    print_rank_0("🩺 Testing NeoX PPO model...")
    config = TRLConfig.load_yaml("configs/neox_ppo_config.yml")
    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # Tokenizer needs to be built in training in order to set the padding vocab
    test_specs(neox_args, config)
