"""
Generates positive movie reviews by tuning a pretrained on IMDB model
with a sentiment reward function.

Usage:
    python gpt-neox/deepy.py examples/neox_ppo_sentiment.py -d configs neox_ppo_model_config.yml
"""
import os

import trlx
from trlx.data.configs import TRLConfig
from trlx.utils.megatron import enable_neox

enable_neox("gpt-neox")

import torch
from datasets import load_dataset
from megatron.neox_arguments import NeoXArgs
from transformers import pipeline


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="configs/neox_ppo_config.yml")
    # parser.add_argument("--neox_args_path", type=str, default="configs/neox_ppo_pythia_19m.yml")
    # parser.add_argument("--gpt_neox_path", type=str, default="gpt-neox/")
    # args = parser.parse_args()

    config_path = "configs/neox_ppo_config.yml"
    config = TRLConfig.load_yaml(config_path)
    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1
    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )
    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples):
        outputs = sentiment_fn(samples, return_all_scores=True)
        sentiments = [output[1]["score"] for output in outputs]
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    # Load pretrained model
    model = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        config=config,
        neox_args=neox_args,
    )
