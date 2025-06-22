# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
from mmlu import sample_generated_mmlu_requests
from shared_prefix import sample_generated_shared_prefix_requests
from simulation import generate_multi_turn_trace, generate_system_prompts_trace
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from sglang.bench_serving import (
    AIOHTTP_TIMEOUT,
    RequestFuncInput,
    RequestFuncOutput,
    calculate_metrics,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
    sample_sharegpt_requests,
)

global args


async def call_generate(session, api_url, text, max_new_tokens, lora_path):
    payload = {
        "text": text,
        "sampling_params": {"max_new_tokens": max_new_tokens},
        "lora_path": lora_path,
    }
    headers = {"Authorization": ""}

    generated_text = ""
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    if chunk == "[DONE]":
                        pass
                    else:
                        data = json.loads(chunk)

                        # NOTE: Some completion API might have a last
                        # usage summary response without a token so we
                        # want to check a token was generated
                        if data["text"]:
                            # generated_text += data["choices"][0]["text"]
                            generated_text += data["text"]

    except Exception:
        raise ValueError("error")

    return generated_text


@dataclass
class ExtraInfo:
    model_id: str
    api_url: str
    extra_request_body: Dict[str, Any]


def generate_thinking_time(mean=1.0):
    """
    生成用户思考时间，采样自指数分布。
    :param mean: 指数分布的平均值（即1/λ）
    :return: 采样得到的思考时间（单位：秒）
    """
    return np.random.exponential(mean)


# set ignore_eos True by default
async def async_request_multi_turn(
    args, request: Dict[str, Any], extra_info: ExtraInfo, pbar: Optional[tqdm] = None
) -> None:
    qas = request["qas"]
    lora_path = request["lora_path"]
    api_url = extra_info.api_url
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        s = ""
        s += qas["system_prompt"]
        qas = qas["qas"]
        for i, qa in enumerate(qas):
            await asyncio.sleep(args.think_time)
            s += qa["prompt"]
            s += await call_generate(
                session=session,
                api_url=api_url,
                text=s,
                max_new_tokens=qa["new_tokens"],
                lora_path=lora_path,
            )
    if pbar:
        pbar.update(1)

    output = RequestFuncOutput()
    output.success = True
    return output


# set ignore_eos True by default
async def async_request_openai_completions(
    args,
    request: Dict[str, Any],
    extra_info: ExtraInfo,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    prompt = request["prompt"]
    lora_path = request["lora_path"]
    prompt_len = request["prompt_len"]
    max_new_tokens = request["max_new_tokens"]
    model = extra_info.model_id
    api_url = extra_info.api_url
    extra_request_body = extra_info.extra_request_body
    # assert api_url.endswith(
    #     "completions"
    # ), "OpenAI Completions API URL must end with 'completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        if args.base_only:
            payload = {
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_new_tokens},
            }
        else:
            payload = {
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_new_tokens},
                "lora_path": lora_path,
            }
        headers = {"Authorization": ""}

        output = RequestFuncOutput()
        output.prompt_len = prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["text"]:
                                # if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                # generated_text += data["choices"][0]["text"]
                                generated_text += data["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    # output.output_len = request_func_input.max_new_tokens
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "system_prompt": async_request_openai_completions,
    "multi_turn": async_request_multi_turn,
    "mmlu": async_request_openai_completions,
    "sharegpt": async_request_openai_completions,
}


def generate_multi_turn_requests(
    args,
    tokenizer,
    num_adapters,
    alpha,
    req_rate,
    cv,
    duration,
    seed=42,
):
    def gen_prompt(tokenizer, token_num):
        all_available_tokens = list(tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        ret = tokenizer.decode(selected_tokens)
        return ret

    def get_cache_path(args):
        # Create cache directory under ~/.cache/sglang
        cache_dir = Path.home() / ".cache" / "sglang"

        # Create a unique cache filename based on the arguments that affect generation
        cache_key = f"qa_{args.num_qa}_{args.turns}_{args.system_prompt_len}_{args.len_q}_{args.len_a}_{args.model}.json"
        return cache_dir / cache_key

    def gen_arguments(args, tokenizer):
        cache_path = get_cache_path(args)

        # Try to load from cache first
        if cache_path.exists():
            print(f"Loading cached arguments from {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)

        print("Generating new arguments...")
        # First progress bar for system prompts
        multi_qas = []
        for _ in tqdm(range(args.num_qa), desc="Generating system prompts"):
            multi_qas.append(
                {
                    "system_prompt": gen_prompt(tokenizer, args.system_prompt_len),
                    "qas": [],
                }
            )

        # Nested progress bars for QA pairs
        for i in tqdm(range(args.num_qa), desc="Generating QA pairs"):
            qas = multi_qas[i]["qas"]
            for j in range(args.turns):
                qas.append(
                    {
                        "prompt": gen_prompt(tokenizer, args.len_q),
                        "new_tokens": args.len_a,
                    }
                )

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(multi_qas, f)
        print(f"Cached arguments saved to {cache_path}")

        return multi_qas

    multi_qas = gen_arguments(args, tokenizer)

    # np.random.seed(seed)
    tot_req = int(req_rate * duration)
    # print(req_rate, duration, tot_req)
    # # generate adapter id
    # probs = np.random.power(alpha, tot_req)
    # ind = (probs * num_adapters).astype(int)
    # # output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
    # # generate timestamp
    requests = []
    # tic = 0
    # shape = 1 / (cv * cv)
    # scale = cv * cv / req_rate
    # # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    # intervals = np.random.gamma(shape, scale, tot_req)
    # adapter_indices = {i: 0 for i in range(num_adapters)}
    # print(ind, tot_req)

    trace = generate_multi_turn_trace(
        adapter_num=num_adapters,
        alpha=float(alpha),
        system_prompt_num=4,
        multi_turn_num=args.turns,
        think_time=args.think_time,
        req_num=tot_req,
        duration=duration,
    )

    pre_ts = 0
    for ts, adapter_id, prompt_id in trace:
        requests.append(
            {
                "interval": ts - pre_ts,
                "adapter_id": adapter_id,
                "qas": multi_qas[prompt_id],
                "lora_path": f"lora{adapter_id}",
            }
        )
        pre_ts = ts
    return requests

    # for i in range(tot_req):
    #     adapter_id = ind[i]
    #     requests.append(
    #         {
    #             "interval": intervals[i],
    #             "adapter_id": adapter_id,
    #             "qas": multi_qas[adapter_indices[adapter_id]],
    #             "lora_path": f"lora{adapter_id}",
    #         }
    #     )
    #     adapter_indices[adapter_id] = adapter_indices[adapter_id] + 1
    # return requests


def get_access_cache_path(num_adapters, alpha, req_rate, cv, duration):
    """Create cache directory under ~/.cache/sglang/benchmark"""
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    # Create a unique cache filename based on the generation parameters
    cache_key = (
        f"generate_system_prompt_requests_{alpha}_{req_rate}_"
        f"{cv}_{args.gsp_question_len}_{num_adapters}_{duration}"
        f".pkl"
    )
    return cache_dir / cache_key


def generate_system_prompt_requests(
    args,
    tokenizer,
    num_adapters,
    alpha,
    req_rate,
    cv,
    duration,
    seed=42,
):
    prompts = []
    max_new_tokens = []
    prompt_lens = []
    input_requests = sample_generated_shared_prefix_requests(
        args=args,
        num_groups=args.gsp_num_groups,
        prompts_per_group=1,
        system_prompt_len=args.gsp_system_prompt_len,
        question_len=args.gsp_question_len,
        output_len=args.gsp_output_len,
        tokenizer=tokenizer,
    )
    for req in input_requests:
        prompts.append(req[0])
        prompt_lens.append(req[1])
        max_new_tokens.append(req[2])

    # cache_path = get_access_cache_path(
    #     num_adapters=num_adapters,
    #     alpha=alpha,
    #     req_rate=req_rate,
    #     cv=cv,
    #     duration=duration,
    # )
    # Try to load from cache first
    # if cache_path.exists():
    #     print(f"\nLoading cached generated input data from {cache_path}")
    #     with open(cache_path, "rb") as f:
    #         tot_req = int(req_rate * duration)
    #         ind =  pickle.load(f)
    #         print(ind, tot_req)
    # else:
    # np.random.seed(seed)
    tot_req = int(req_rate * duration)
    # # generate adapter id
    # probs = np.random.power(alpha, tot_req)
    # ind = (probs * num_adapters).astype(int)
    # # generate timestamp
    requests = []
    # tic = 0
    # shape = 1 / (cv * cv)
    # scale = cv * cv / req_rate
    # # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    # intervals = np.random.gamma(shape, scale, tot_req)
    # adapter_indices = {i: 0 for i in range(num_adapters)}
    # # ind = [39, 62, 54 ,49 ,25 ,25 ,15 ,59 ,49 ,53  ,9 ,63 ,58 ,29 ,27 ,27 ,35 ,46 ,42 ,34 ,50 ,23 ,34 ,38, 43 ,56 ,28 ,45 ,49 ,13 ,49 ,26 ,16 ,62 ,62 ,57 ,35 ,20 ,52 ,42 ,22 ,45 ,11 ,61 ,32 ,52 ,35 ,46, 47 ,27 ,63 ,56 ,62 ,60 ,49 ,61 ,19 ,28 ,13 ,36 ,39 ,33 ,58 ,38 ,33 ,47 ,24 ,57 ,17 ,63 ,56 ,28, 4 ,57 ,53 ,54 ,56 ,17 ,38 ,21 ,59 ,50 ,36 ,16 ,35 ,36 ,54 ,51 ,60 ,43 ,22 ,54 ,55 ,47 ,56 ,44,46 ,41 ,10 ,21 ,11 ,51 ,35 ,45 ,60 ,31 ,40 ,55 ,30 ,17 ,34 ,25 ,61 ,57 ,50 ,59 ,57 ,27 ,60 ,47,57 ,60 ,36 ,21 ,30 ,41 ,57 ,59  ,5 ,45 ,41 ,30 ,22 ,37 ,62 ,36 ,46 ,53 ,38 ,63 ,62 ,32 ,45 ,35,34 ,12 ,49 ,45 ,14 ,33 ,60 ,31 ,24 ,44 ,63 ,31 ,52 ,55 ,31 ,54 ,38 ,50 ,50 ,46 ,19 ,58 ,36 ,27,12 ,49 ,52  ,8 ,45 ,30 ,51 ,26 ,53 ,39 ,61 ,23 ,37 ,21 ,61 ,59 ,32 ,51 ,57 ,47 ,46 ,31 ,19 ,60,60 ,50 ,37 ,37 ,54 ,60 ,60 ,56 ,51 ,18 ,25 ,60 ,49 , 6 ,20 ,52  ,4 ,25 ,47 ,53 ,51 ,30 ,54 ,31,36 ,55 ,51 ,58 ,51 ,48 ,19 ,38 ,32 ,31 ,63 ,40 ,60 ,50 ,57 ,45 ,48 ,44 ,28 ,54 ,33 , 9 ,51 ,26,62 ,62 ,61 ,38  ,7 ,61 ,41 ,62 ,62 ,59 ,34 ,39 ,59, 36 ,26 ,47 ,61 ,53 ,48 ,19 ,50 ,63 ,23 ,46,59 ,55 ,53 ,53 ,38 ,34 ,57 ,57 ,59 ,61 ,45 ,45 ,57 ,51 ,53 ,57 ,60 ,37 ,39 ,19 ,48 ,12 ,43 ,47,34 ,49 ,11 ,12 ,58 ,38 ,22 ,46 ,56 ,29 ,50 ,18 ,14 ,46 ,47 ,51 ,54 ,63 ,45 ,36 ,57 ,33 ,42 ,17,10 ,62 ,58 ,53 ,40 ,26 ,25 ,32 ,47 ,54 ,52 ,33 ,62 ,54 ,47 ,50 ,41 ,31 ,38 ,55  ,7 ,21 ,13 ,12,59 ,53 ,44 ,20 ,44 ,44 ,26 ,42 ,40 ,50 ,51 ,13, 39 ,50 ,45 ,59 ,51 ,25 ,17 ,51 ,10 ,48 ,62 ,48]
    # # tot_req = 255
    # print(ind, tot_req)

    trace = generate_system_prompts_trace(
        adapter_num=num_adapters,
        alpha=float(alpha),
        system_prompt_num=4,
        req_num=tot_req,
        duration=duration,
    )

    # cache_path.parent.mkdir(parents=True, exist_ok=True)
    # print(f"Caching generated input data to {cache_path}")
    # with open(cache_path, "wb") as f:
    #     pickle.dump(ind, f)

    pre_ts = 0
    for ts, adapter_id, prompt_id in trace:
        requests.append(
            {
                "interval": ts - pre_ts,
                "adapter_id": adapter_id,
                "prompt": prompts[prompt_id],
                "max_new_tokens": max_new_tokens[prompt_id],
                "prompt_len": prompt_lens[prompt_id],
                "lora_path": f"lora{adapter_id}",
            }
        )
        pre_ts = ts
    return requests

    # for i in range(tot_req):
    #     adapter_id = ind[i]
    #     requests.append(
    #         {
    #             "interval": intervals[i],
    #             "adapter_id": adapter_id,
    #             "prompt": prompts[adapter_indices[adapter_id]],
    #             "max_new_tokens": max_new_tokens[adapter_indices[adapter_id]],
    #             "prompt_len": prompt_lens[adapter_indices[adapter_id]],
    #             "lora_path": f"lora{adapter_id}",
    #         }
    #     )
    #     adapter_indices[adapter_id] = adapter_indices[adapter_id] + 1
    # return requests


def generate_mmlu_requests(
    args,
    tokenizer,
    num_adapters,
    alpha,
    req_rate,
    cv,
    duration,
    seed=42,
):
    prompts = []
    max_new_tokens = []
    arguments, prompt_lens, labels, num_questions = sample_generated_mmlu_requests(
        args=args,
    )
    for argument in arguments:
        prompts.append(argument["examples"] + argument["question"])

    for _ in prompts:
        max_new_tokens.append(10)

    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_adapters).astype(int)

    # output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)

    adapter_indices = {i: 0 for i in range(num_adapters)}

    print(ind, tot_req)

    for i in range(tot_req):
        adapter_id = ind[i]
        requests.append(
            {
                "interval": intervals[i],
                "adapter_id": adapter_id,
                "prompt": prompts[adapter_indices[adapter_id]],
                "max_new_tokens": max_new_tokens[adapter_indices[adapter_id]],
                "prompt_len": prompt_lens[adapter_indices[adapter_id]],
                "lora_path": f"lora{adapter_id}",
            }
        )
        adapter_indices[adapter_id] = adapter_indices[adapter_id] + 1
    return requests


def generate_sharegpt_requests(
    args,
    tokenizer,
    num_adapters,
    alpha,
    req_rate,
    cv,
    duration,
    seed=42,
):
    prompts = []
    max_new_tokens = []
    prompt_lens = []

    input_requests = sample_sharegpt_requests(
        dataset_path=args.dataset_path,
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
        fixed_output_len=args.sharegpt_output_len,
        context_len=args.sharegpt_context_len,
        prompt_suffix=args.prompt_suffix,
        apply_chat_template=args.apply_chat_template,
    )
    for req in input_requests:
        prompts.append(req[0])
        prompt_lens.append(req[1])
        max_new_tokens.append(req[2])

    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_adapters).astype(int)

    # output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)

    adapter_indices = {i: 0 for i in range(num_adapters)}

    print(ind, tot_req)

    for i in range(tot_req):
        adapter_id = ind[i]
        requests.append(
            {
                "interval": intervals[i],
                "adapter_id": adapter_id,
                "prompt": prompts[adapter_indices[adapter_id]],
                "max_new_tokens": max_new_tokens[adapter_indices[adapter_id]],
                "prompt_len": prompt_lens[adapter_indices[adapter_id]],
                "lora_path": f"lora{adapter_id}",
            }
        )
        adapter_indices[adapter_id] = adapter_indices[adapter_id] + 1
    return requests


GENERATOR_REQUEST_FUNCS = {
    "system_prompt": generate_system_prompt_requests,
    "multi_turn": generate_multi_turn_requests,
    "mmlu": generate_mmlu_requests,
    "sharegpt": generate_sharegpt_requests,
}


async def get_request(
    requests: List[Dict[str, Any]],
) -> AsyncGenerator[Dict[str, Any], None]:
    requests = iter(requests)
    for request in requests:
        yield request

        # Sample the request interval from the exponential distribution.
        interval = request["interval"]
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float


def calculate_metrics(
    input_requests: List[Dict[str, Any]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    benchmark: str,
    backend: str,
) -> Tuple[BenchmarkMetrics, List[int]]:
    output_lens: List[int] = []
    retokenized_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_len
            output_lens.append(output_len)
            retokenized_output_len = len(
                tokenizer.encode(outputs[i].generated_text, add_special_tokens=False)
            )
            retokenized_output_lens.append(retokenized_output_len)
            if benchmark != "multi_turn":
                prompt_len = input_requests[i]["prompt_len"]
                total_input += prompt_len
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)

            e2e_latencies.append(outputs[i].latency)

            completed += 1
        else:
            output_lens.append(0)
            retokenized_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(output_lens),
        total_output_retokenized=sum(retokenized_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(output_lens) / dur_s,
        output_throughput_retokenized=sum(retokenized_output_lens) / dur_s,
        total_throughput=(total_input + sum(output_lens)) / dur_s,
        total_throughput_retokenized=(total_input + sum(retokenized_output_lens))
        / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        p95_itl_ms=np.percentile(itls or 0, 95) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        max_itl_ms=np.max(itls or 0) * 1000,
        mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
        median_e2e_latency_ms=np.median(e2e_latencies) * 1000,
        std_e2e_latency_ms=np.std(e2e_latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(e2e_latencies, 99) * 1000,
        concurrency=np.sum(e2e_latencies) / dur_s,
    )

    return metrics, output_lens


async def benchmark(
    benchmark: str,
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    requests: List[Dict[str, Any]],
    disable_tqdm: bool,
    extra_request_body: Dict[str, Any],
):
    if benchmark in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[benchmark]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    extra_info = ExtraInfo(
        api_url=api_url,
        model_id=model_id,
        extra_request_body=extra_request_body,
    )

    print("Starting initial single prompt test run...")
    request = requests[0]
    test_output = await request_func(args=args, request=request, extra_info=extra_info)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    pbar = None if disable_tqdm else tqdm(total=len(requests))

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []

    async for request in get_request(requests):
        tasks.append(
            asyncio.create_task(
                request_func(
                    args=args, request=request, extra_info=extra_info, pbar=pbar
                )
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, output_lens = calculate_metrics(
        input_requests=requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        benchmark=benchmark,
        backend=backend,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", metrics.total_output_retokenized
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format("Total throughput (tok/s):", metrics.total_throughput)
    )
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            "backend": args.backend,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "output_throughput": metrics.output_throughput,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            "duration": benchmark_duration,
            "completed": metrics.completed,
        }
    else:
        print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"

    # Append results to a JSONL file
    with open(output_file_name, "a") as file:
        file.write(json.dumps(result) + "\n")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
    }
    return result


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set url
    if args.port is None:
        args.port = {
            "sglang": 30000,
        }.get(args.backend, 30000)

    # api_url = (
    #     f"{args.base_url}/v1/completions"
    #     if args.base_url
    #     else f"http://{args.host}:{args.port}/v1/completions"
    # )
    api_url = (
        f"{args.base_url}/generate"
        if args.base_url
        else f"http://{args.host}:{args.port}/generate"
    )

    print(f"{args}\n")

    file_path = "/u/cjia/sglang-common/sglang/cache_lora_benchmark/acc.txt"
    with open(file_path, "r+") as f:
        f.truncate(0)

    # Read dataset
    backend = args.backend
    benchmark_name = args.benchmark
    model_id = args.model
    tokenizer_id = args.model

    tokenizer = get_tokenizer(tokenizer_id)

    num_adapters = args.num_adapters
    alpha = args.alpha
    cv = args.cv
    req_rate = args.req_rate
    duration = args.duration

    if benchmark_name in GENERATOR_REQUEST_FUNCS:
        generate_func = GENERATOR_REQUEST_FUNCS[benchmark_name]

    requests = generate_func(
        args=args,
        tokenizer=tokenizer,
        num_adapters=num_adapters,
        alpha=alpha,
        req_rate=req_rate,
        cv=cv,
        duration=duration,
    )

    return asyncio.run(
        benchmark(
            benchmark=benchmark_name,
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            requests=requests,
            disable_tqdm=False,
            extra_request_body={},
        )
    )


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online lora serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        default="sglang",
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        default="default",
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--num-adapters",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--req-rate",
        type=float,
        default=3,
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--think-time",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random dataset.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
    )

    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--nsub", type=int, default=60)

    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--num-qa", type=int, default=128)
    parser.add_argument("--system-prompt-len", type=int, default=1024)
    parser.add_argument("--len-q", type=int, default=16)
    parser.add_argument("--len-a", type=int, default=64)

    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default="",
        help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template",
    )

    # for
    parser.add_argument(
        "--gsp-num-groups",
        type=int,
        default=32,
        help="Number of system prompt groups for generated-shared-prefix dataset",
    )
    parser.add_argument(
        "--gsp-prompts-per-group",
        type=int,
        default=16,
        help="Number of prompts per system prompt group for generated-shared-prefix dataset",
    )
    parser.add_argument(
        "--gsp-system-prompt-len",
        type=int,
        default=1024,
        help="Target length in tokens for system prompts in generated-shared-prefix dataset",
    )
    parser.add_argument(
        "--gsp-question-len",
        type=int,
        default=64,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    parser.add_argument(
        "--gsp-output-len",
        type=int,
        default=128,
        help="Target length in tokens for outputs in generated-shared-prefix dataset",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    args = parser.parse_args()
    run_benchmark(args)
