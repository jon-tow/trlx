"""
Usage:
accelerate launch examples/multinode/benchmark.py --config examples/multinode/configs/ppo_multinode.yml
accelerate launch examples/multinode/benchmark.py --config examples/multinode/configs/ilql_multinode.yml
"""
import argparse
import os
from typing import *

import torch
import yaml
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.configs import TRLConfig


# Benchmark: Sentiment RL
# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def ilql_sentiment(config):
    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )

    def metric_fn(samples: List[str]) -> Dict[str, List[float]]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return {"sentiments": sentiments}

    imdb = load_dataset("imdb", split="train+test")

    trlx.train(
        dataset=(imdb["text"], imdb["label"]),
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        metric_fn=metric_fn,
        config=config,
    )


def ppo_sentiment(config):
    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )
    print("DONE WITH PIPELINE")

    def reward_fn(samples: List[str]) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    print("LOADING DATASET")
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    print("BEGIN TRAINING")
    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="examples/benchmark/configs/ppo.yml"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    config = yaml.safe_load(open(args.config))
    config = TRLConfig.from_dict(config)
    if config.method.name == "ilqlconfig":
        ilql_sentiment(config)
    elif config.method.name == "ppoconfig":
        ppo_sentiment(config)
    else:
        raise ValueError("Unknown method name")
