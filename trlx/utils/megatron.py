import copy
import time
from typing import *

import torch
import torch.nn.functional as F
from megatron import mpu, print_rank_0
from megatron.model.gpt2_model import GPT2ModelPipe
from megatron.neox_arguments.arguments import NeoXArgs
from megatron.text_generation_utils import (
    broadcast_terminate_signal,
    filter_logits,
    get_batch,
    pad_batch,
    stop_tokens_in_completion,
    switch,
)
from megatron.utils import is_mp_rank_0


def enable_neox(neox_path):
    """Enable gpt-neox support.
    HF Accelerate tries to import megatron-lm, but it's incompatible with gpt-neox's megatron
    """
    from accelerate.utils import imports

    def always_false():
        return False

    imports.is_megatron_lm_available = always_false

    import sys

    sys.path.append(neox_path)

    # Import models to register them for get_model
    import megatron

    print(f"using megatron from {megatron.__file__}")
    from trlx.model.neox_ppo_model import NeoXPPOModel
    from trlx.orchestrator.neox_ppo_orchestrator import NeoXPPOOrchestrator


# Megatron generation utils


def forward_model(model, model_inputs, is_pipe_parallel=False) -> torch.Tensor:
    """Overriden from `megatron.text_generation_utils.forward_model`"""
    # Pipeline modules couldn't use kwargs, we need to forward a pipe model
    # differently to a normal model.
    if not is_pipe_parallel:
        return model.module(model_inputs)
    else:
        # We need to format inputs this way because:
        # a) deepspeed pipeline only accepts iterables
        # b) deepspeed pipeline *requires* that you pass in labels for the loss, it's not easy to get around this
        # so we wrap the inputs in an iterable, and pad them (because internally, we get labels as inputs[:, 1:] and inputs as inputs[:, :-1])
        model_inputs = iter([{"text": F.pad(model_inputs[0], pad=(0, 1))}])

        # Set num microbatches to 1 at inference time
        micro_batches_before = model.micro_batches
        model.micro_batches = 1

        # Deepspeed sends metadata across pipeline stages only once in the first step, then assumes it will stay
        # constant. In inference, the metadata of the tensors being sent across pipe stages may change, so we need to set
        # these two flags in order for deepspeed to send the metadata every step, otherwise torch.distributed hangs
        # silently. Fun stuff.
        model.first_output_send = True
        model.pipe_recv_buf = None

        loss, logits = model.eval_batch(model_inputs, return_logits=True)
        model.micro_batches = micro_batches_before
        return logits


def stream_tokens(
    neox_args,
    model,
    context_tokens: List[List[int]],
    eos_token_id: int = None,
    maximum_tokens: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_tokens=None,
):
    """Overridden from megatron.text_generation_utils.stream_tokens to handle multi-head models"""
    model.eval()

    # pad batch in order to allow conversion to tensor
    context_tokens, context_lengths = pad_batch(
        copy.deepcopy(context_tokens),
        pad_id=neox_args.tokenizer.eod,
        pad_len=neox_args.seq_length,
    )

    # convert to tensor and broadcast
    context_tokens = torch.cuda.LongTensor(context_tokens)
    if stop_tokens:
        if len(stop_tokens) > 0 and type(stop_tokens[0]) is not list:
            stop_tokens = [stop_tokens]
        for i in range(0, len(stop_tokens)):
            stop_tokens[i] = torch.cuda.LongTensor(stop_tokens[i])

    # Make sure context tokens + start tokens are the same across all ranks
    token_generation_start_index = torch.cuda.LongTensor(context_lengths)
    torch.distributed.broadcast(
        context_tokens,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    torch.distributed.broadcast(
        token_generation_start_index,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )

    # get attention mask / position ids
    context_tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens)

    # set variables
    eos_token_id = eos_token_id or neox_args.tokenizer.eod
    maximum_tokens = maximum_tokens or (
        neox_args.seq_length - token_generation_start_index.max().item() - 1
    )
    batch_size = context_tokens.size(0)

    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate
    last_token_index_to_generate = min(
        neox_args.seq_length
        - 1,  # never generate more than the model's sequence length
        token_index_to_generate + maximum_tokens - 1,
    )

    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().cuda()
        token_generation_end_index = torch.ones([batch_size]).long().cuda() * (-1)
        generation_logits = (
            torch.empty(maximum_tokens, neox_args.padded_vocab_size).float().cuda()
        )

        while token_index_to_generate <= last_token_index_to_generate:
            if recompute:  # recompute all tokens
                model_inputs = (
                    context_tokens,
                    position_ids,
                    attention_mask,
                )
                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = logits[
                        :, token_index_to_generate - 1, :
                    ]  # [bs, seq, vocab_size] -> [bs, vocab_size]
            else:  # use kv cache
                if token_index_to_generate == first_token_index_to_generate:
                    tokens_to_use = context_tokens[:, :token_index_to_generate]
                    positions_to_use = position_ids[:, :token_index_to_generate]
                else:
                    tokens_to_use = context_tokens[:, token_index_to_generate - 1].view(
                        batch_size, -1
                    )
                    positions_to_use = position_ids[
                        :, token_index_to_generate - 1
                    ].view(batch_size, -1)

                model_inputs = (
                    tokens_to_use,  # input_ids
                    positions_to_use,  # position_ids
                    attention_mask,  # attention_mask
                )

                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = (
                        logits[:, -1].view(batch_size, -1).contiguous()
                    )  # [bs, seq, vocab_size] -> [bs, vocab_size]

            if logits is not None:
                # sample token id of the to be generated token
                if temperature == 0.0 and top_k == 0 and top_p == 0.0:
                    generated_tokens = torch.argmax(
                        generated_token_logits, dim=-1
                    ).view(-1)
                else:
                    generated_token_logits = generated_token_logits.float()
                    if temperature > 0.0:
                        generated_token_logits /= temperature
                    generated_token_logits = filter_logits(
                        generated_token_logits, top_k=top_k, top_p=top_p
                    )
                    next_token_log_probs = F.softmax(generated_token_logits, dim=-1)
                    generated_tokens = torch.multinomial(
                        next_token_log_probs, num_samples=1
                    ).view(-1)

                if neox_args.return_logits:
                    generation_logits[
                        token_index_to_generate - 1
                    ] = generated_token_logits[0]

            if neox_args.is_pipe_parallel:
                # broadcast generated tokens to pipe parallel group
                src_rank = model.grid.stage_to_global(model.num_stages - 1)
                generated_tokens = (
                    generated_tokens
                    if logits is not None
                    else torch.zeros(batch_size, dtype=torch.long).cuda()
                )
                torch.distributed.broadcast(
                    tensor=generated_tokens,
                    src=src_rank,
                    group=mpu.get_pipe_parallel_group(),
                )

            # determine if state has started for each batch item
            state_started = (
                token_generation_start_index <= token_index_to_generate
            )  # check which batch items have been started

            # switch out padding tokens for generated tokens
            context_tokens[:, token_index_to_generate] = switch(
                context_tokens[:, token_index_to_generate].view(-1),
                generated_tokens,
                state_started,
            )

            # determine if state has finished for each batch item
            state_done = (
                generated_tokens == eos_token_id
            ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
            state_just_finished = (state_done & ~state_is_done).bool()
            state_is_done = state_is_done | state_done
            stop_tokens_produced = torch.zeros_like(state_is_done)
            for batch_idx, ctx in enumerate(context_tokens):
                stop_tokens_produced[batch_idx] = stop_tokens_in_completion(
                    stop_tokens, context_tokens, batch_idx, token_index_to_generate
                )
            state_is_done = state_is_done | stop_tokens_produced

            token_generation_end_index[
                (state_started.byte() & ~state_is_done).bool()
            ] = token_index_to_generate

            token_index_to_generate += 1

            yield context_tokens, token_generation_start_index, token_generation_end_index, generation_logits, state_is_done.bool()
            if torch.all(state_is_done):
                break


def generate(
    neox_args: NeoXArgs,
    model: GPT2ModelPipe,
    input_ids: torch.Tensor,
    context_lengths: List[int],
    eos_token_id: Optional[int] = None,
    # TODO: Add `min_length` support.
    min_length: Optional[int] = 64,
    max_length: Optional[int] = 64,
    recompute: Optional[bool] = True,
    temperature: Optional[float] = 0.0,
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 0.0,
    stop_tokens: Optional[torch.Tensor] = None,
    do_sample: Optional[bool] = False,
    return_logits: Optional[bool] = False,
    **kwargs,
):
    """Generates samples from tokens.

    TODO: This can be optimized:
    - Currently it generates one output at a time but `stream_tokens` should
        support batched generation.
    - Recompute (caching) support does not currently work - this should also help.

    Args:
        input_ids: The sequence used as a prompt for the generation.
        context_lengths: The length of the prompt for each input. This is used to
            strip the padding tokens from the input.
            TODO: Remove this hack
        eos_token_id: The end of text token at which completion is terminated, even
            if max_tokes count has not been reached
        max_new_tokens: The maximum number of okens to be generated
        recompute: Flag indicating whether a cache is used for already forwarded
            tokens (true) or whether all tokens are recomputed at every iteration
            (false)
        temperature (default 0.0): exponential scaling output distribution ("higher
            == more risk")
        top_k (default 0): integer -> integer between 0 and the models vocab size.
            Filters out any logits with a probability less than that of the top_kth
            token.
        top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the
            smallest possible set of tokens whose cumulative probability exceeds
            the probability top_p.
            NOTE: greedy decoding is used if temperature is 0.0, top_k is 0 and
            top_p is 0.0
    """
    input_count = len(input_ids)
    input_pos = 0
    generations = []
    while True:
        # model.module.clear_cache()  # clear kv cache between batches
        # start_time = time.time()
        terminate_runs = 0
        if input_pos == input_count:
            terminate_runs = 1
        else:
            context_length = context_lengths[input_pos]
            # print("="* 80)
            # print_rank_0(f"context_tokens: {input_ids[input_pos]}")
            context_tokens = input_ids[input_pos][:context_length]
            # print_rank_0(f"input_pos: {input_pos}, context_length: {context_length}")
            # print_rank_0(f"truncated context_tokens: {context_tokens}")
            # print("="* 80)
            if context_length >= (neox_args.seq_length // 2):
                print_rank_0(f"Warning: context length {context_length} is too long")
            input_pos += 1

        if not is_mp_rank_0():
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)
            terminate_runs = 0

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs:
            break

        # print_rank_0("=" * 40)
        # print_rank_0('Context:', neox_args.tokenizer.detokenize(context_tokens))
        # print_rank_0('Tokens:', context_tokens)
        # print_rank_0("=" * 40)

        for (
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_generated_token_logits,
            is_done,
        ) in stream_tokens(
            neox_args=neox_args,
            model=model,
            context_tokens=[context_tokens.tolist()],
            eos_token_id=eos_token_id,
            maximum_tokens=max_length,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=stop_tokens,
        ):
            pass  # Finish generation and use all results below

        batch_context_tokens = batch_context_tokens.cpu().numpy().tolist()
        batch_token_generation_start_index = (
            batch_token_generation_start_index.cpu().numpy().tolist()
        )
        batch_token_generation_end_index = (
            batch_token_generation_end_index.cpu().numpy().tolist()
        )
        batch_is_done = is_done.cpu().numpy().tolist()

        for tokens, start_index, end_index, is_done in zip(
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_is_done,
        ):
            if end_index >= start_index:
                generated_tokens = tokens[
                    start_index : end_index + 1
                ]  # tokens[start_index : end_index + 1]
            else:
                generated_tokens = []
                # This will happen if the first generated token is a stop token or eos token
                # message = "WARNING: text generation did not start; try different batching or adjust parameters"
            if is_mp_rank_0():
                # TODO: Possibly return more informative metadata:
                data = generated_tokens
                # data = {
                #     "query": context_tokens,
                #     "response": torch.tensor(generated_tokens),
                #     "sample": torch.tensor(all_tokens),
                # }
                # data = {
                # "context": text,
                # "text": generated_text,
                # "length": len(generated_tokens),
                # "finished": is_done,
                # "message": message,
                # "duration_seconds": float(time.time() - start_time),
                # }
                # if neox_args.return_logits:
                #     data["logits"] = batch_generated_token_logits.cpu().numpy().tolist()
                # print_rank_0("=" * 40)
                # c = input_ids[input_pos][:context_length]
                # c = context_tokens
                # print_rank_0('Context Text:', neox_args.tokenizer.detokenize(context_tokens.tolist()))
                # print_rank_0('Context Tokens:', c)
                # print_rank_0('Generated Text:', neox_args.tokenizer.detokenize(generated_tokens))
                # print_rank_0('Generated Tokens:', generated_tokens)
                # print_rank_0("=" * 40)
                generations.append(data)
    return generations
