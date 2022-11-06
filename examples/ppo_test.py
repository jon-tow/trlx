# Generates positive movie reviews by tuning a pretrained on IMDB model
# with a sentiment reward function
# import argparse
# import sys
import trlx

from datasets import load_dataset
from transformers import pipeline
from trlx.data.configs import TRLConfig

# Force ignore `accelerate`'s Megatron-LM support
from accelerate.utils import imports
imports.is_megatron_lm_available = False


TEST_CONFIG = TRLConfig.load_yaml("configs/ppo_test.yml")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--neox-path", type=str, default="gpt-neox")
    # args = parser.parse_args()
    # sys.path.append(args.neox_path)

    sentiment_fn = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=-1)

    def reward_fn(samples):
        outputs = sentiment_fn(samples, return_all_scores=True)
        sentiments = [output[1]["score"] for output in outputs]
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    model = trlx.train(
        "lvwerra/gpt2-imdb",
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        config=TEST_CONFIG
    )
