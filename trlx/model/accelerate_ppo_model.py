import torch
from torchtyping import TensorType
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch
from trlx.model import register_model
from trlx.model.accelerate_base_model import AccelerateRLModel
from trlx.model.nn.ppo_models import ( # isort:skip
    AdaptiveKLController,
    FixedKLController,
    GPTHydraHeadWithValueModel,
)
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.utils.modeling import logprobs_from_logits
from typing import Tuple


@register_model
class AcceleratePPOModel(AccelerateRLModel):
    def __init__(self, config):
        super().__init__(config)

        self.store = PPORolloutStorage(self.tokenizer.pad_token_id)

        rollout_loader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(
                config.method.init_kl_coef, config.method.target, config.method.horizon
            )
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def get_arch(self, config: TRLConfig):
        return GPTHydraHeadWithValueModel(
            self.config.model.model_path, self.config.model.num_layers_unfrozen
        )

    def get_model_inputs(
        self,
        query_tensors: TensorType["batch_size", "query_size"],
        response_tensors: TensorType["batch_size", "response_size"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.cat((query_tensors, response_tensors), dim=1)
        attention_mask = (
            tokens.not_equal(self.tokenizer.pad_token_id)
            .long()
            .to(tokens.device)
        )
        # For a proper positional encoding in case of left padding
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        return tokens, attention_mask, position_ids

    def loss(self, batch: PPORLBatch):
        # Move `batch` data to accelerator device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        ref_logprobs = batch.logprobs.to(self.accelerator.device)
        ref_values = batch.values.to(self.accelerator.device)
        ref_rewards = batch.rewards.to(self.accelerator.device)

        # Compute advantages and returns
        response_length = response_tensors.shape[1]
        advantages, returns = self.config.method.get_advantages_and_returns(
            ref_values, ref_rewards, response_length
        )
        
        # Extract logprobs and value predictions from the model
        # NOTE: Need to do this here because the model is class specific.
        tokens, attention_mask, position_ids = self.get_model_inputs(
            query_tensors, response_tensors)
        logits, _, values_pred = self.model(
            tokens, attention_mask, position_ids=position_ids)
        logprobs = logprobs_from_logits(logits[:, :-1, :], tokens[:, 1:])
        # Only the response part of the values/logprobs is needed
        logprobs, values_pred, mask = (
            logprobs[:, -response_length:],
            values_pred[:, -response_length:],
            attention_mask[:, -response_length:],
        )

        loss, stats = self.config.method.loss(
            advantages=advantages,
            returns=returns,
            logprobs=logprobs,
            values_pred=values_pred,
            ref_logprobs=ref_logprobs,
            ref_values=ref_values,
            mask=mask,
        )

        # Update kl controller stats
        self.approx_kl = stats["policy/approx_kl"]

        return loss, stats

    def post_epoch_callback(self):
        self.store.clear_history()
        self.orch.make_experience(
            self.config.method.num_rollouts, self.iter_count
        )  # Collect more rollouts for training

    def post_backward_callback(self):
        # Update kl_coefficient
        self.kl_ctl.update(self.approx_kl, n_steps=self.config.train.batch_size)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        train_dataloader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            train_dataloader, eval_dataloader
        )

        self.n_updates_per_batch = self.config.method.ppo_epochs
        self.total_steps = (
            self.config.train.epochs
            * self.n_updates_per_batch
            * len(self.train_dataloader)
        )
        self.total_steps = min(self.total_steps, self.config.train.total_steps)
