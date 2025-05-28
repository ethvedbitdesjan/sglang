import argparse
import asyncio
import json
import os
import pickle
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)


def gen_prompt(tokenizer, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def get_gen_prefix_cache_path(args, tokenizer):
    """Create cache directory under ~/.cache/sglang/benchmark"""
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    # Create a unique cache filename based on the generation parameters
    cache_key = (
        f"gen_shared_prefix_{args.gsp_num_groups}_{args.gsp_prompts_per_group}_"
        f"{args.gsp_system_prompt_len}_{args.gsp_question_len}_{args.gsp_output_len}_"
        f"{tokenizer.__class__.__name__}.pkl"
    )
    return cache_dir / cache_key


def sample_generated_shared_prefix_requests(
    args,
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    """Generate benchmark requests with shared system prompts using random tokens and caching."""
    cache_path = get_gen_prefix_cache_path(args, tokenizer)

    # Try to load from cache first
    if cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("\nGenerating new input data...")

    # Generate system prompts for each group
    system_prompts = []
    for _ in range(num_groups):
        system_prompt = gen_prompt(tokenizer, system_prompt_len)
        system_prompts.append(system_prompt)

    # Generate questions
    questions = []
    for _ in range(num_groups * prompts_per_group):
        question = gen_prompt(tokenizer, question_len)
        questions.append(question)

    # Combine system prompts with questions
    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0

    for group_idx in tqdm(range(num_groups), desc="Generating system prompt"):
        system_prompt = system_prompts[group_idx]
        for prompt_idx in tqdm(
            range(prompts_per_group), desc="Generating questions", leave=False
        ):
            question = questions[group_idx * prompts_per_group + prompt_idx]
            full_prompt = f"{system_prompt}\n\n{question}"
            prompt_len = len(tokenizer.encode(full_prompt))

            input_requests.append((full_prompt, prompt_len, output_len))
            total_input_tokens += prompt_len
            total_output_tokens += output_len

    # Shuffle questions
    random.shuffle(input_requests)

    print(input_requests[:5])

    # Print statistics
    print(f"\nGenerated shared prefix dataset statistics:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Total prompts: {len(input_requests)}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(
        f"Average system prompt length: {sum(len(tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
    )
    print(
        f"Average question length: {sum(len(tokenizer.encode(q)) for q in questions) / len(questions):.1f} tokens\n"
    )

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Caching generated input data to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(input_requests, f)

    return input_requests
