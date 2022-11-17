from typing import Callable
from megatron import print_rank_0
from megatron.text_generation_utils import get_batch

import torch

# from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.model.neox_ppo_model import NeoXPPOModel
from trlx.model.nn.ppo_models import GPTHeadWithValueModel, GPTHydraHeadWithValueModel
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits


@register_orchestrator
class NeoXPPOOrchestrator(Orchestrator):
    """
    Orchestrator that prepares data for PPO training: transforms samples from `pipeline` into `PPOBatch` and pushes them into model's `store`
    """

    def __init__(
        self,
        model: NeoXPPOModel,
        pipeline: BasePipeline,
        reward_fn: Callable,
        metric_fn: Callable = None,
        chunk_size: int = 512,
    ):
        self.pipeline = pipeline
        self.rl_model = model
        self.chunk_size = chunk_size

        self.pipeline_loader = self.pipeline.create_loader(
            self.chunk_size, shuffle=True
        )
        self.pipeline_iterator = iter(self.pipeline_loader)

        if not hasattr(self.rl_model.model, "frozen_head"):
            self.ref_model = self.rl_model.get_arch(self.rl_model.config)

        self.rl_model.orch = self
        self.rl_model.reward_fn = reward_fn
        self.rl_model.metric_fn = metric_fn

    def score(self, samples):
        """
        Batched scoring function taking text and generating scalar
        """
        return self.rl_model.reward_fn(samples)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model, computes KL againts a reference model appends PPOElements to model's `store`
        """
        ppo_rl_elements = []
        stats = {}
        clock = Clock()
        while len(ppo_rl_elements) < num_rollouts:

            # Get the next batch in the prompt dataset and refresh if exhausted
            try:
                batch = next(self.pipeline_iterator)
            except StopIteration:
                self.pipeline_iterator = iter(self.pipeline_loader)
                batch = next(self.pipeline_iterator)

            # Split batch into query and response tokens and merge as samples
            zipped_batch = zip(batch["input_ids"], batch["context_lengths"])
            query_tensors = [input[:length].tolist() for input, length in zipped_batch]
            response_tensors = self.rl_model.generate(**batch)
            samples = [q + r for q, r in zip(query_tensors, response_tensors)]

            # Generate scores
            texts = self.rl_model.tokenizer.batch_decode(
                samples, skip_special_tokens=True
            )
            scores = torch.as_tensor(self.score(texts))

            # Construct model inputs
            all_tokens = self.rl_model.tokenizer.pad(samples)["input_ids"]
            # Replace with call to self.rl_model.get_model_inputs(all_tokens)
            all_tokens, attention_mask, position_ids = get_batch(
                self.rl_model.neox_args, all_tokens
            )

            # Precompute logprobs, values
            with torch.no_grad():
                # TODO(jon-tow): Model currently only returns logits!
                # Deepspeed requires the ordering: (input_ids, position_ids, attention_mask)
                logits, _, v = self.rl_model.model(
                    (all_tokens, position_ids, attention_mask)
                )
                breakpoint()
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if hasattr(self.rl_model.model, "frozen_head"):
                    ref_logits = self.rl_model.model.forward_hydra(
                        all_tokens, return_dict=False
                    )
                else:
                    ref_logits, _, _ = self.ref_model(
                        (all_tokens, position_ids, attention_mask)
                    )
            logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logits = ref_logits.to(self.rl_model.model.device)
            ref_logprobs = logprobs_from_logits(
                ref_logits[:, :-1, :], all_tokens[:, 1:]
            )

            start = query_tensors.size()[1] - 1
            end = query_tensors.size()[1] + response_tensors.size()[1] - 1
            all_values = v[:, start:end]
            all_logprobs = logprobs[:, start:end]
            all_ref_logprobs = ref_logprobs[:, start:end]

            # Compute rewards
            kls = all_logprobs - all_ref_logprobs
            non_score_rewards = -self.rl_model.kl_ctl.value * kls
            all_rewards = non_score_rewards.clone()
            all_rewards[:, -1] += scores.to(torch.cuda.current_device())

            # Move to CPU
            query_tensors = query_tensors.cpu()
            response_tensors = response_tensors.cpu()
            all_logprobs = all_logprobs.cpu()
            all_values = all_values.cpu()
            all_rewards = all_rewards.cpu()

            exp_time = clock.tick()

            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_tensors[i, :],
                    response_tensor=response_tensors[i, :],
                    logprobs=all_logprobs[i, :],
                    values=all_values[i, :],
                    rewards=all_rewards[i, :],
                )
                for i in range(query_tensors.size()[0])
            ]
            ppo_rl_elements += new_ppo_rl_elements

        # TODO: Add logging
        stats = {"exp_time": exp_time}
        # self.rl_model.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to model's rollout storage
        self.rl_model.push_to_store(ppo_rl_elements)
