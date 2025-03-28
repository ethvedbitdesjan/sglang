import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import tiktoken
from tqdm import tqdm

# Configuration
BASE_URL = "http://127.0.0.1:30000/generate"

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def load_mmlu_data(data_dir, nsub):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    # Build prompts
    arguments = []
    labels = []
    num_questions = []

    for subject in subjects[:nsub]:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[:5]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )
        num_questions.append(test_df.shape[0])

        k = 5
        few_shot_examples = gen_prompt(dev_df, subject, k)
        while len(tokenizer.encode(few_shot_examples)) > 1536:
            k -= 1
            few_shot_examples = gen_prompt(dev_df, subject, k)

        for i in range(test_df.shape[0]):
            prompt_end = format_example(test_df, i, include_answer=False)

            arguments.append(few_shot_examples + prompt_end)

            label = test_df.iloc[i, test_df.shape[1] - 1]
            labels.append(label)

    return arguments, labels, num_questions


def gen_prompt(train_df, subject, k):
    prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {df.iloc[idx, j+1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k+1]}\n\n"
    return prompt


def generate_requests(prompts, num_adapters, alpha, req_rate, cv, duration, seed=42):
    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_adapters).astype(int)

    print(ind)

    # output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)

    adapter_indices = {i: 0 for i in range(num_adapters)}

    for i in range(tot_req):
        tic += intervals[i]
        adapter_id = ind[i]
        requests.append(
            {
                "timestamp": tic,
                "adapter_id": adapter_id,
                "data": prompts[adapter_indices[adapter_id]],
                "lora_path": f"lora{adapter_id}",
            }
        )
        adapter_indices[adapter_id] = adapter_indices[adapter_id] + 1
    return requests


def send_requests(requests):
    start_time = time.time()
    last_sent = 0

    for i, req in enumerate(requests):
        # Calculate wait time
        current_real_time = time.time() - start_time
        wait_time = req["timestamp"] - current_real_time

        if wait_time > 0:
            time.sleep(wait_time)

        # Build payload
        payload = {
            "text": [req["data"]],
            # "sampling_params": {"max_new_tokens": 1},
            "lora_path": [req["lora_path"]],
        }

        # Send request
        try:
            response = requests.post(BASE_URL, json=payload, timeout=30)
            status = response.status_code
        except Exception as e:
            status = str(e)

        # Print status
        if i % 100 == 0:
            tqdm.write(
                f"Sent {i+1}/{len(requests)} "
                f"[Adapter {req['adapter_id']}] Status: {status}"
            )


def main(args):
    # Load MMLU data
    arguments, labels, num_questions = load_mmlu_data(args.data_dir, args.nsub)

    N_ADAPTERS = 5
    ALPHA = 2
    CV = 1
    REQ_RATE = 3
    duration = 20

    # Schedule requests
    requests = generate_requests(
        texts=arguments,
        num_adapters=N_ADAPTERS,
        alpha=ALPHA,
        req_rate=REQ_RATE,
        cv=CV,
        duration=duration,
    )

    print(requests)

    # Send requests
    # send_requests(requests)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/u/cjia/sglang-common/data/data"
    )
    parser.add_argument("--nsub", type=int, default=5, help="Number of subjects to use")
    parser.add_argument(
        "--duration", type=int, default=15, help="Test duration in seconds"
    )
    args = parser.parse_args()

    main(args)
