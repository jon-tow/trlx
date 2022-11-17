import os
from typing import Callable, Iterable, List, Optional, Tuple

from accelerate import Accelerator

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import AccelerateILQLModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator

# Manually import NeoX PPO modules to force register
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.utils.loading import get_model, get_orchestrator


def train(
    model_path: Optional[str] = None,
    reward_fn: Optional[Callable] = None,
    dataset: Optional[Iterable[Tuple[str, float]]] = None,
    prompts: Optional[List[str]] = None,
    eval_prompts: Optional[List[str]] = None,
    metric_fn: Optional[Callable] = None,
    config: Optional[TRLConfig] = None,
    neox_args: Optional[dict] = None,
    split_token: Optional[str] = None,
    logit_mask: Optional[List[List[bool]]] = None,
):
    """
    Dispatches online or offline reinforcement training depending on whether a reward function or a list of samples & rewards is given

    Args:
        model_path (Optional[str]): Path to either huggingface checkpoint or a local directory
        reward_fn (List[str] -> List[float]): Function to rate batches of generated samples
        dataset (List[str], List[float]): Lists of samples and rewards
        prompts (List[str]): Prompts to sample off from during online training
        eval_prompts (List[str]): Prompts to periodically validate training on
        metric_fn (Optional[Callable[List[str], List[float]]]): Function to compute statistics on validation samples
        split_token (Optional[str]): Split samples in the dataset on prompts and continuations
        logit_mask (Optional[List]): Bigram masking matrix
    """

    if reward_fn is not None:
        if config is None:
            config = TRLConfig.load_yaml("configs/ppo_config.yml")

        # Create the model
        if model_path:
            config.model.model_path = model_path
        model_class = get_model(config.model.model_type)
        # TODO: Find a better pattern for this initialization
        if neox_args is not None:
            from megatron.utils import print_rank_0

            print_rank_0("Initializing model with NeoX args")
            print_rank_0(f"NeoX Args:\n{neox_args}")
            model = model_class(config, neox_args)
        else:
            model = model_class(config)

        # Extract the prompts to be used
        batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        prompts = prompts or [model.tokenizer.bos_token] * batch_size
        if eval_prompts is None:
            eval_prompts = prompts[:batch_size]

        # Create prompt pipelines
        pipeline_tokenizer = (
            model.tokenizer
        )  # None # model.tokenizer if isinstance(model, NeoXRLModel) else None
        pipeline = PromptPipeline(prompts, pipeline_tokenizer)
        # Create orchestrator
        orch = get_orchestrator(config.train.orchestrator)(
            model, pipeline, reward_fn=reward_fn, chunk_size=config.method.chunk_size
        )
        # Orchestrate (generate) the rollouts and push to storage
        orch.make_experience(config.method.num_rollouts)

        # Add the evaluation pipeline
        eval_pipeline = PromptPipeline(eval_prompts, pipeline_tokenizer)
        model.add_eval_pipeline(eval_pipeline)

    elif dataset is not None:
        samples, rewards = dataset

        if len(samples) != len(rewards):
            raise ValueError(
                f"Number of samples {len(samples)} should match the number of rewards {len(rewards)}"
            )

        if config is None:
            config = TRLConfig.load_yaml("configs/ilql_config.yml")

        if model_path:
            config.model.model_path = model_path

        model = AccelerateILQLModel(
            config=config,
            logit_mask=logit_mask,
            metric_fn=metric_fn,
        )

        batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        if eval_prompts is None:
            eval_prompts = [model.tokenizer.bos_token] * batch_size

        eval_pipeline = PromptPipeline(eval_prompts, model.tokenizer)

        orch = OfflineOrchestrator(model, split_token=split_token)
        orch.make_experience(samples, rewards)
        model.add_eval_pipeline(eval_pipeline)

    else:
        raise ValueError(f"Either {dataset=} or {reward_fn=} should be given")

    model.learn()
    return model
