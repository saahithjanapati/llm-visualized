#!/usr/bin/env python3
"""Interactive utility for extracting GPT-2 activation traces."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from gpt_extraction.quantization import QuantizationMode
from gpt_extraction.trace_collector import (
    GPT2TraceCollector,
    TraceConfig,
    parameter_checkpoints,
    save_trace,
)


@dataclass
class CompletionCandidate:
    text: str
    new_token_ids: List[int]
    step_logs: List[Dict[str, object]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_logits(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Return a filtered view of ``logits`` according to top-k / top-p rules."""

    scores, indices = torch.sort(logits, descending=True)

    if top_k and top_k > 0:
        scores = scores[:top_k]
        indices = indices[:top_k]

    if top_p < 1.0:
        probs = torch.softmax(scores, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative > top_p
        if sorted_indices_to_remove.any():
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            keep_mask = ~sorted_indices_to_remove
            scores = scores[keep_mask]
            indices = indices[keep_mask]

    return torch.stack([indices, scores], dim=-1)


def sample_sequence(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    tokenizer: GPT2TokenizerFast,
) -> CompletionCandidate:
    device = input_ids.device
    generated = input_ids.unsqueeze(0)
    past_key_values = None
    new_tokens: List[int] = []
    step_logs: List[Dict[str, object]] = []

    for step in range(max_new_tokens):
        if past_key_values is None:
            model_inputs = generated
        else:
            model_inputs = generated[:, -1:]

        outputs = model(
            input_ids=model_inputs,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :].squeeze(0) / temperature
        past_key_values = outputs.past_key_values

        filtered = filter_logits(logits, top_k=top_k, top_p=top_p)
        candidate_ids = filtered[:, 0].to(dtype=torch.long)
        candidate_logits = filtered[:, 1]
        probs = torch.softmax(candidate_logits, dim=-1)
        next_index = torch.multinomial(probs, num_samples=1)
        next_token = candidate_ids[next_index]

        generated = torch.cat([generated, next_token.view(1, 1)], dim=-1)
        new_tokens.append(int(next_token))

        candidate_ids_list = candidate_ids.tolist()
        candidate_logits_list = candidate_logits.tolist()
        step_logs.append(
            {
                "step": step,
                "token_id": int(next_token),
                "token": tokenizer.decode([int(next_token)], clean_up_tokenization_spaces=False),
                "candidates": [
                    {
                        "token_id": int(tok),
                        "logit": float(logit),
                    }
                    for tok, logit in zip(candidate_ids_list, candidate_logits_list)
                ],
            }
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(new_tokens, clean_up_tokenization_spaces=False)
    return CompletionCandidate(text=text, new_token_ids=new_tokens, step_logs=step_logs)


def generate_candidates(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    input_ids: torch.Tensor,
    *,
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> List[CompletionCandidate]:
    candidates = []
    for _ in range(num_return_sequences):
        candidates.append(
            sample_sequence(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                tokenizer=tokenizer,
            )
        )
    return candidates


def prompt_interactive_selection(
    prompt_text: str,
    candidates: Sequence[CompletionCandidate],
    tokenizer: GPT2TokenizerFast,
) -> CompletionCandidate:
    print("\n--- Prompt ---")
    print(prompt_text)
    print("\n--- Candidate completions ---")
    for idx, candidate in enumerate(candidates):
        preview = candidate.text.replace("\n", "\\n")
        print(f"[{idx}] {preview}")

    while True:
        choice = input("Select a completion index, or 'm' to enter your own: ").strip()
        if choice.lower() in {"m", "manual"}:
            manual = input("Enter custom completion text: ")
            tokens = tokenizer.encode(manual, add_special_tokens=False)
            return CompletionCandidate(text=manual, new_token_ids=tokens, step_logs=[])
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(candidates):
                candidate = candidates[idx]
                truncate = input("Truncate completion to N tokens (press enter to keep all): ").strip()
                if truncate:
                    try:
                        n = int(truncate)
                        candidate = CompletionCandidate(
                            text=tokenizer.decode(candidate.new_token_ids[:n], clean_up_tokenization_spaces=False),
                            new_token_ids=candidate.new_token_ids[:n],
                            step_logs=candidate.step_logs[:n],
                        )
                    except ValueError:
                        print("Invalid number, keeping full completion.")
                return candidate
        print("Please enter a valid option.")


def build_trace_document(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt_text: str,
    prompt_ids: List[int],
    completion: CompletionCandidate,
    *,
    collector: GPT2TraceCollector,
    config: TraceConfig,
    generation_args: Dict[str, object],
) -> Dict[str, object]:
    full_tokens = prompt_ids + completion.new_token_ids
    trace_payload = collector.collect(torch.tensor(full_tokens, dtype=torch.long))

    document = {
        "meta": {
            "model": model.config._name_or_path,
            "quantization": config.quantization.value,
            "stride": config.stride,
            "head_stride": config.head_step(),
            "mlp_stride": config.mlp_step(),
            "generation": generation_args,
        },
        "prompt": {
            "text": prompt_text,
            "token_ids": prompt_ids,
            "tokens": tokenizer.convert_ids_to_tokens(prompt_ids),
        },
        "completion": {
            "text": completion.text,
            "token_ids": completion.new_token_ids,
            "tokens": tokenizer.convert_ids_to_tokens(completion.new_token_ids),
            "steps": completion.step_logs,
        },
        "sequence": {
            "token_ids": full_tokens,
            "tokens": tokenizer.convert_ids_to_tokens(full_tokens),
            "text": tokenizer.decode(full_tokens, clean_up_tokenization_spaces=False),
        },
        "activations": trace_payload["activations"],
        "flops": trace_payload["flops"],
        "parameters": parameter_checkpoints(model),
    }
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GPT-2 activation traces")
    parser.add_argument("--model", default="gpt2", help="Model name or path")
    parser.add_argument("--prompt", help="Optional prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--num-return-sequences", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--head-stride", type=int)
    parser.add_argument("--mlp-stride", type=int)
    parser.add_argument(
        "--quantization",
        choices=[mode.value for mode in QuantizationMode],
        default=QuantizationMode.FLOAT16.value,
    )
    parser.add_argument("--output", default="gpt2_trace.json")
    parser.add_argument("--parameters-output", help="Optional file to dump parameter summary")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--manual-completion", help="Skip sampling and use this completion text")
    parser.add_argument("--truncate-to", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    prompt_text = args.prompt or input("Enter prompt: ")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
    }

    if args.manual_completion is not None:
        completion = CompletionCandidate(
            text=args.manual_completion,
            new_token_ids=tokenizer.encode(args.manual_completion, add_special_tokens=False),
            step_logs=[],
        )
    else:
        candidates = generate_candidates(
            model,
            tokenizer,
            input_tensor,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        if args.no_interactive:
            completion = candidates[0]
        else:
            completion = prompt_interactive_selection(prompt_text, candidates, tokenizer)

        if args.truncate_to is not None:
            completion = CompletionCandidate(
                text=tokenizer.decode(completion.new_token_ids[: args.truncate_to], clean_up_tokenization_spaces=False),
                new_token_ids=completion.new_token_ids[: args.truncate_to],
                step_logs=completion.step_logs[: args.truncate_to],
            )

    config = TraceConfig(
        stride=args.stride,
        head_stride=args.head_stride,
        mlp_stride=args.mlp_stride,
        quantization=QuantizationMode(args.quantization),
    )
    collector = GPT2TraceCollector(model, config)

    document = build_trace_document(
        model,
        tokenizer,
        prompt_text,
        prompt_ids,
        completion,
        collector=collector,
        config=config,
        generation_args=generation_args,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_trace(str(output_path), document)
    print(f"Trace written to {output_path}")

    if args.parameters_output:
        params = parameter_checkpoints(model)
        with open(args.parameters_output, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print(f"Parameter summary written to {args.parameters_output}")


if __name__ == "__main__":
    main()

