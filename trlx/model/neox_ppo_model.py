from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import *

import deepspeed
import megatron
from megatron.model.gpt2_model import GPT2ModelPipe
from megatron.neox_arguments.arguments import NeoXArgs
from megatron.text_generation_utils import stream_tokens
import torch
from megatron import mpu, print_rank_0
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.tokenizer.tokenizer import HFTokenizer
from megatron.training import (
    get_batch_pipe,
    get_learning_rate_scheduler,
    get_optimizer,
    setup_model_and_optimizer,
)
from megatron.utils import Timers, get_ltor_masks_and_position_ids, get_total_params
from torchtyping import TensorType
from transformers.tokenization_utils_base import BatchEncoding

from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch
from trlx.model import BaseRLModel, register_model
from trlx.model.nn.ppo_models import (
    AdaptiveKLController,
    FixedKLController,
    GPTHydraHeadWithValueModel,
    GPTNeoXHeadWithValueModel,
)
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.utils.modeling import logprobs_from_logits
from trlx.utils.megatron import generate as megatron_generate

EncodedInput = List[int]


class NeoXTokenizerWrapper(object):
    """Wraps NeoX HFTokenizer to make it compatible with HuggingFace's"""

    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eod
        self.eos_token_id = tokenizer.eod_id
        # NeoX use eos_token_id as bos_token_id
        self.bos_token = tokenizer.eod
        self.bos_token_id = tokenizer.eod_id
        self.pad_token_id = tokenizer.pad_id

    def decode(self, token_ids: torch.LongTensor, skip_special_tokens: bool = False):
        # TODO(jon-tow): Implement `skip_special_tokens`
        return self.tokenizer.tokenizer.decode(token_ids)

    def batch_decode(self, sequences: torch.Tensor, skip_special_tokens: bool = False):
        # TODO(jon-tow): Implement `skip_special_tokens`
        return [self.decode(sequence) for sequence in sequences]

    def encode(self, text: str):
        return self(text)

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """Overrides the default HuggingFace tokenizer pad method.
        NOTE: This does not properly return `BatchEncoding` objects; fix this if needed.
        NOTE: `return_tensors` is ignored and forced to PyTorch tensors
        TODO: Generalize this to work with all types of inputs - only supports List[List[int]].
        """
        max_length = max_length or max(len(x) for x in encoded_inputs)
        padded_inputs = defaultdict(list)
        for encoded_input in encoded_inputs:
            padding_length = max_length - len(encoded_input)
            padded_inputs["input_ids"].append(
                # NOTE: Use `eos_token` here to match `megatron.text_generation_utils.pad_batch`
                encoded_input
                + [self.eos_token] * padding_length
            )
            padded_inputs["context_lengths"].append(len(encoded_input))
        padded_inputs["input_ids"] = torch.tensor(padded_inputs["input_ids"])
        return padded_inputs

    def __call__(self, text: str) -> torch.Tensor:
        return self.tokenizer.tokenizer.encode(text).ids


@register_model
class NeoXRLModel(BaseRLModel):
    """RL Model that uses NeoX for training"""

    def __init__(
        self,
        config: TRLConfig,
        neox_args: megatron.NeoXArgs,
        train_mode: Optional[bool] = True,
        init_loggers: Optional[bool] = False,
    ):
        super().__init__(config, train_mode)

        # Initialize and get arguments, timers, and wandb.
        self.neox_args = neox_args
        if init_loggers:
            megatron.utils.init_wandb(neox_args=neox_args)
            self.timers = Timers(
                use_wandb=neox_args.use_wandb,
                tensorboard_writer=neox_args.tensorboard_writer,
            )
        megatron.initialize.initialize_megatron(neox_args)

        # Initialize the model, optimizer, and learning rate scheduler.
        self.model, self.opt, self.scheduler = self.setup_model_and_optimizer(
            neox_args, config
        )
        breakpoint()
        self.tokenizer = NeoXTokenizerWrapper(neox_args.tokenizer)

    def get_model(self, neox_args: NeoXArgs, config: TRLConfig) -> torch.nn.Module:
        print_rank_0("⏳ Loading the model...")
        model = GPTNeoXHeadWithValueModel(neox_args, config)
        if not neox_args.is_pipe_parallel:
            # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
            model = model.to_sequential()
        if neox_args.deepspeed:
            # DeepSpeed handles CUDA, FP16, and DDP components.
            return model
        else:
            raise ValueError("Must be using deepspeed to run neox")

    def setup_model_and_optimizer(self, neox_args, config, iteration=None):
        model = self.get_model(neox_args, config)
        opt, param_groups = get_optimizer(model, neox_args)
        scheduler = get_learning_rate_scheduler(opt, neox_args)
        if neox_args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")
            if neox_args.no_load_optim:
                assert opt is None
                _model_params = None
                _scheduler = None
            else:
                _model_params = param_groups if opt is None else None
                _scheduler = scheduler

            model, opt, _, scheduler = deepspeed.initialize(
                model=model,
                optimizer=opt,
                args=neox_args,
                lr_scheduler=_scheduler,
                dist_init_required=True,
                model_parameters=_model_params,
                config_params=neox_args.deepspeed_config,
                mpu=mpu if not neox_args.is_pipe_parallel else None,
            )
            model.total_params = get_total_params(model.module)
            print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

            if neox_args.is_pipe_parallel:
                model.set_has_attention_mask(True)
                # NOTE: Ignore b/c we don't use gpt-neox data loading.
                # model.set_batch_fn(partial(get_batch_pipe, neox_args=neox_args))
        else:
            raise ValueError("Must be using deepspeed to run neox")
        if neox_args.load is not None:
            print_rank_0(
                f"⏳ Loading checkpoint and starting from iteration {neox_args.iteration}"
            )
            neox_args.iteration = load_checkpoint(
                neox_args=neox_args,
                model=model,
                optimizer=opt,
                lr_scheduler=scheduler,
                iteration=iteration,
            )
        else:
            neox_args.iteration = 0
        return model, opt, scheduler

    def tokenize(self, text: Union[Sequence[str], Sequence[torch.LongTensor]]):
        """
        Tokenize a batch of text after adding bos token to each of the samples
        """
        if isinstance(text[0], torch.LongTensor):
            return text
        text = [self.tokenizer.bos_token + txt for txt in text]
        # TODO: Add truncation - this might not be batched?
        # return self.tokenizer(
        #     text,
        #     truncation=True,
        #     max_length=self.config.seq_length,
        #     return_tensors="pt",
        # )
        return torch.tensor([self.tokenizer(t) for t in text])

    def generate(
        self,
        input_ids: torch.LongTensor,
        context_lengths: List[int],
        attention_mask: torch.LongTensor = None,
        use_cache: bool = False,
        **kwargs,
    ):
        self.model.module.inference_mode(use_cache=use_cache)
        # input_ids = input_ids["input_ids"].to(torch.cuda.current_device())
        # if attention_mask is not None:
        #    attention_mask = attention_mask.to(torch.cuda.current_device())
        kwargs = dict(self.generate_kwargs, **kwargs)
        with torch.no_grad():
            return megatron_generate(
                neox_args=self.neox_args,
                model=self.model,
                input_ids=input_ids,
                context_lengths=context_lengths,
                **kwargs,
            )

    def save(self, directory=None):
        """Creates checkpoint of optimizer, scheduler and a model"""
        self.accelerator.save_state(directory or self.config.train.checkpoint_dir)
        save_checkpoint(
            neox_args=self.neox_args,
            iteration=self.iter_count,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    @abstractmethod
    def get_arch(self, config: TRLConfig):
        """Returns a specific wrapper of the decoder architecture"""
        pass

    def get_components(self) -> Dict[str, Any]:
        components = (
            {"model": self.model, "opt": self.opt, "scheduler": self.scheduler}
            if self.train_mode
            else {"model": self.model}
        )
        return components

    @abstractmethod
    def loss(self, batch) -> Tuple[float, Dict]:
        """Compute loss on a batch from `store` and return some statistics"""
        pass

    @abstractmethod
    def post_backward_callback(self):
        """Do something after model update"""
        pass

    @abstractmethod
    def post_epoch_callback(self):
        """Do something after exhausting/single pass over `self.store`"""
        pass


@register_model
class NeoXPPOModel(NeoXRLModel):
    def __init__(
        self,
        config: TRLConfig,
        neox_args: megatron.NeoXArgs,
    ):
        super().__init__(config, neox_args)

        self.store = PPORolloutStorage(self.tokenizer.pad_token_id)
        # TODO: Prepare `rollout_loader` for distributed training with deepspeed-megatron,
        # if required.
        rollout_loader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
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
            # TODO: Clean this up. Maybe just pull the ids from neox_args?
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def get_arch(self, config: TRLConfig):
        return GPTHydraHeadWithValueModel(
            self.config.model.model_path, self.config.model.num_layers_unfrozen
        )

    # def get_model_inputs(
    #     self,
    #     query_tensors: TensorType["batch_size", "query_size"],
    #     response_tensors: TensorType["batch_size", "response_size"],
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     # Move to GPU
    #     tokens = torch.cat((query_tensors, response_tensors), dim=1)
    #     tokens = tokens.contiguous().cuda()
    #     # Get the attention mask and position ids
    #     attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
    #         data=tokens,
    #         eod_token=self.neox_args.tokenizer.eod,
    #         eod_mask_loss=self.neox_args.eod_mask_loss,
    #     )
    #     return tokens, attention_mask, position_ids

    def loss(self, batch: PPORLBatch):
        # TODO: Implement in megatron
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)

        response_length = response_tensors.shape[-1]
        advantages, returns = self.config.method.get_advantages_and_returns(
            old_values, old_rewards, response_length
        )

        tokens, attention_mask, position_ids = self.get_model_inputs(
            query_tensors, response_tensors
        )
        logits, _, values_pred = self.model(
            tokens, attention_mask, position_ids=position_ids
        )
        logprobs = logprobs_from_logits(logits[:, :-1, :], tokens[:, 1:])
        # Only the response part of the values/logprobs is needed
        logprobs, values_pred, mask = (
            logprobs[:, -response_length:],
            values_pred[:, -response_length:],
            attention_mask[:, -response_length:],
        )

        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )
        self.approx_kl = stats["policy/approx_kl"]  # Update kl controller stats
        return loss, stats

    def post_epoch_callback(self):
        self.store.clear_history()
        self.orch.make_experience(
            self.config.method.num_rollouts, self.iter_count
        )  # Collect more rollouts for training

    def post_backward_callback(self):
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
