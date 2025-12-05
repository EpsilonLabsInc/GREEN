import re
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
import os
import json
from tqdm import tqdm
import numpy as np
import time
import sys
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import necessary functions (ensure these are available in your environment)
from green_score.utils import (
    gather_processes,
    make_prompt,
    clean_responses,
    compute_largest_cluster,
    flatten_values_lists_of_list_dicts_to_dict,
)

from transformers.utils import logging

# Set the logging level for the transformers library to ERROR to suppress warnings that have been resolved
logging.get_logger("transformers").setLevel(logging.ERROR)


def load_azure_config(config_path=None):
    """Load Azure OpenAI configuration from config.json file."""
    if config_path is None:
        # Look for config.json in the project root (parent of green_score)
        config_path = Path(__file__).parent.parent / "config.json"

    if not Path(config_path).exists():
        return None

    with open(config_path, "r") as f:
        config = json.load(f)

    return config.get("azure_openai", None)


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def load_config(config_path=None):
    """Load configuration from config.json and config.secret.json files."""
    if config_path is None:
        # Look for config.json in the project root (parent of green_score)
        config_path = Path(__file__).parent.parent / "config.json"

    if not Path(config_path).exists():
        return None, None, False

    with open(config_path, "r") as f:
        config = json.load(f)

    azure_config = config.get("azure_openai", {})
    batch_size = config.get("batch_size", 8)
    parallel = config.get("parallel", False)

    # Load secrets from config.secret.json
    secret_path = Path(config_path).parent / "config.secret.json"
    if secret_path.exists():
        with open(secret_path, "r") as f:
            secret_config = json.load(f)
        # Merge secret config into azure_config
        if "azure_openai" in secret_config:
            azure_config.update(secret_config["azure_openai"])

    return azure_config if azure_config else None, batch_size, parallel


def load_azure_config(config_path=None):
    """Load Azure OpenAI configuration from config.json and config.secret.json files."""
    azure_config, _, _ = load_config(config_path)
    return azure_config


def tqdm_on_main(*args, **kwargs):
    if is_main_process():
        print("==== Beginning Inference ====")
        return tqdm(*args, **kwargs)
    else:
        return kwargs.get("iterable", None)


class GREEN:
    def __init__(
        self,
        model_name=None,
        output_dir=".",
        cpu=False,
        compute_summary_stats=True,
        use_azure=False,
        azure_config_path=None,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        if self.verbose:
            print("[VERBOSE] Initializing GREEN scorer...")
        warnings.filterwarnings(
            "ignore", message="A decoder-only architecture is being used*"
        )
        self.cpu = cpu
        self.output_dir = output_dir

        # Load batch_size from config
        _, config_batch_size, config_parallel = load_config(azure_config_path)
        self.batch_size = config_batch_size
        self.parallel = config_parallel
        if self.verbose:
            print(f"[VERBOSE] Batch size: {self.batch_size}")
            print(f"[VERBOSE] Parallel requests: {self.parallel}")

        self.max_length = 2048
        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]
        self.prompts = None
        self.completions = None
        self.green_scores = None
        self.error_counts = None

        # Azure OpenAI support
        self.use_azure = use_azure
        self.azure_client = None
        self.azure_deployment = None

        if use_azure:
            if self.verbose:
                print("[VERBOSE] use_azure=True, initializing Azure OpenAI client...")
            self._init_azure_client(azure_config_path)
        elif (
            torch.cuda.is_available() and torch.cuda.device_count() > 1 and not self.cpu
        ):
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                )
                torch.cuda.set_device(dist.get_rank())
                if dist.get_rank() == 0:
                    print(
                        "Distributed training with", torch.cuda.device_count(), "GPUs"
                    )
        self.model = None
        self.tokenizer = None
        if model_name and not use_azure:
            self.model_name = model_name.split("/")[-1]
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=False if "Phi" in model_name else True,
                device_map=(
                    {"": "cuda:{}".format(torch.cuda.current_device())}
                    if not self.cpu
                    else {"": "cpu"}
                ),
                torch_dtype=torch.float16,
            )

            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                add_eos_token=True,
                use_fast=True,
                trust_remote_code=True,
                padding_side="left",
            )

            chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

            self.tokenizer.chat_template = chat_template
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.clean_up_tokenization_spaces = True
            assert self.tokenizer.padding_side == "left"

        self.compute_summary_stats = compute_summary_stats

    def _init_azure_client(self, config_path=None):
        """Initialize Azure OpenAI client from config file."""
        if self.verbose:
            print("[VERBOSE] Loading Azure OpenAI configuration...")
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for Azure OpenAI support. "
                "Install it with: pip install openai"
            )

        azure_config = load_azure_config(config_path)
        if self.verbose:
            print(
                f"[VERBOSE] Azure config loaded: endpoint={azure_config.get('endpoint') if azure_config else None}"
            )
        if azure_config is None:
            raise ValueError(
                "Azure OpenAI config not found. Please create a config.json file "
                "in the project root with azure_openai configuration."
            )

        # Allow environment variable to override config file api_key
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", azure_config.get("api_key"))
        if api_key == "<your-api-key>":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Azure OpenAI API key not configured. Set AZURE_OPENAI_API_KEY "
                    "environment variable or update config.json."
                )

        self.model_name = azure_config.get("model_name", "gpt-5.1")
        self.azure_deployment = azure_config.get("deployment", self.model_name)

        self.azure_client = AzureOpenAI(
            api_version=azure_config.get("api_version", "2024-12-01-preview"),
            azure_endpoint=azure_config.get("endpoint"),
            api_key=api_key,
        )

        if self.verbose:
            print(f"[VERBOSE] Azure OpenAI client created successfully")
            print(
                f"[VERBOSE] Model: {self.model_name}, Deployment: {self.azure_deployment}"
            )

        if is_main_process():
            print(f"Initialized Azure OpenAI client with model: {self.model_name}")

    def __call__(self, refs, hyps):
        if self.verbose:
            print(
                f"[VERBOSE] __call__ invoked with {len(refs)} references and {len(hyps)} hypotheses"
            )
        if is_main_process():
            print("Processing data...making prompts")

        dataset = Dataset.from_dict({"reference": refs, "prediction": hyps})
        if self.verbose:
            print(f"[VERBOSE] Dataset created with {len(dataset)} examples")

        dataset = self.process_data(dataset)
        if self.verbose:
            print(f"[VERBOSE] Prompts generated")
        if is_main_process():
            print("Done.")

        self.dataset = dataset

        t = time.time()
        if self.verbose:
            print("[VERBOSE] Starting inference...")

        mean, std, green_scores, summary, results_df = self.infer()

        t = time.time() - t
        if is_main_process():
            print("Seconds per example: ", t / len(refs))

        # Skip distributed cleanup for Azure mode (no distributed training)
        if not self.use_azure and not is_main_process():
            print(f"Rank {dist.get_rank()} exiting.")
            dist.destroy_process_group()
            sys.exit()

        return mean, std, green_scores, summary, results_df

    def process_data(self, dataset):
        def prompting(examples):
            return {
                "prompt": [
                    make_prompt(r, p)
                    for r, p in zip(examples["reference"], examples["prediction"])
                ]
            }

        dataset = dataset.map(prompting, batched=True)
        return dataset

    @torch.inference_mode()
    def infer(self):
        if self.use_azure:
            return self._infer_azure()

        assert self.model is not None and self.tokenizer is not None

        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not self.cpu:
            dataset_dist = split_dataset_by_node(
                self.dataset,
                rank=get_rank(),
                world_size=int(os.environ["WORLD_SIZE"]),
            )
            print("Distributed dataset created on rank: ", int(os.environ["RANK"]))
        else:
            dataset_dist = self.dataset

        local_completions = []
        local_references = []

        for batch in tqdm_on_main(
            iterable=dataset_dist.iter(batch_size=self.batch_size),
            total=len(dataset_dist) // self.batch_size,
        ):
            local_references.extend(batch["prompt"])
            local_completions.extend(self.get_response(batch))

        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not self.cpu:
            self.completions, self.prompts = gather_processes(
                local_completions, local_references
            )
        else:
            self.completions = local_completions
            self.prompts = local_references

        if is_main_process():
            print("==== End Inference ====")

        if len(self.completions) != len(self.prompts):
            print("Length of prompts and completions are not equal!")

        return self.process_results()

    def _infer_azure(self):
        """Run inference using Azure OpenAI API."""
        if self.verbose:
            print("[VERBOSE] _infer_azure started")
        assert self.azure_client is not None, "Azure client not initialized"

        self.completions = []
        self.prompts = []

        total_batches = len(self.dataset) // self.batch_size + (
            1 if len(self.dataset) % self.batch_size else 0
        )
        if self.verbose:
            print(
                f"[VERBOSE] Processing {len(self.dataset)} examples in {total_batches} batches (batch_size={self.batch_size})"
            )

        batch_num = 0
        for batch in tqdm_on_main(
            iterable=self.dataset.iter(batch_size=self.batch_size),
            total=len(self.dataset) // self.batch_size,
        ):
            batch_num += 1
            if self.verbose:
                print(f"[VERBOSE] Processing batch {batch_num}/{total_batches}...")
            self.prompts.extend(batch["prompt"])
            self.completions.extend(self._get_azure_response(batch))
            if self.verbose:
                print(
                    f"[VERBOSE] Batch {batch_num} completed. Total completions: {len(self.completions)}"
                )

        if is_main_process():
            print("==== End Inference ====")

        if len(self.completions) != len(self.prompts):
            print("Length of prompts and completions are not equal!")

        return self.process_results()

    def _get_azure_response(self, batch):
        """Get responses from Azure OpenAI API."""
        assert "prompt" in batch.keys(), "prompt is not in batch keys"

        prompts = batch["prompt"]

        if self.parallel:
            return self._get_azure_response_parallel(prompts)
        else:
            return self._get_azure_response_sequential(prompts)

    def _get_azure_response_sequential(self, prompts):
        """Get responses sequentially."""
        response_list = []
        for idx, prompt in enumerate(prompts):
            if self.verbose:
                print(
                    f"[VERBOSE] Sending request {idx+1}/{len(prompts)} to Azure OpenAI..."
                )
                print(f"[VERBOSE] Prompt length: {len(prompt)} characters")
            try:
                response = self.azure_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_completion_tokens=self.max_length,
                    model=self.azure_deployment,
                    temperature=0,
                )
                if self.verbose:
                    print(f"[VERBOSE] Response received for request {idx+1}")
                response_text = response.choices[0].message.content
                response_text = clean_responses(response_text)
                response_list.append(response_text)
            except Exception as e:
                print(f"Error getting Azure response: {e}")
                response_list.append("")

        return response_list

    def _get_azure_response_parallel(self, prompts):
        """Get responses in parallel using ThreadPoolExecutor."""
        if self.verbose:
            print(f"[VERBOSE] Sending {len(prompts)} requests in parallel...")

        def fetch_single_response(idx_prompt):
            idx, prompt = idx_prompt
            try:
                response = self.azure_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_completion_tokens=self.max_length,
                    model=self.azure_deployment,
                    temperature=0,
                )
                response_text = response.choices[0].message.content
                response_text = clean_responses(response_text)
                return idx, response_text
            except Exception as e:
                print(f"Error getting Azure response for request {idx+1}: {e}")
                return idx, ""

        # Use ThreadPoolExecutor for parallel requests
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = {
                executor.submit(fetch_single_response, (i, p)): i
                for i, p in enumerate(prompts)
            }

            # Use tqdm for progress tracking
            pbar = tqdm(as_completed(futures), total=len(prompts), desc="Parallel requests", disable=not self.verbose)
            for future in pbar:
                idx, response_text = future.result()
                results[idx] = response_text

        if self.verbose:
            print(f"[VERBOSE] All {len(prompts)} parallel requests completed")

        return results

    def tokenize_batch_as_chat(self, batch):
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) if not self.cpu else "cpu"
        batch = [
            self.tokenizer.apply_chat_template(
                i, tokenize=False, add_generation_prompt=True
            )
            for i in batch
        ]

        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(local_rank)

        return batch

    def get_response(self, batch):
        assert "prompt" in batch.keys(), "prompt is not in batch keys"

        batch = [
            [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
            for prompt in batch["prompt"]
        ]

        batch = self.tokenize_batch_as_chat(batch)

        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=2048,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response_list = []
        if isinstance(responses, list):
            for response in responses:
                response = clean_responses(response)
                response_list.append(response)
        else:
            responses = clean_responses(responses)
            response_list.append(responses)

        return response_list

    def process_results(self):
        self.green_scores = [
            self.compute_green(response) for response in self.completions
        ]
        self.error_counts = pd.DataFrame(
            [self.compute_error_count(response) for response in self.completions],
            columns=self.sub_categories + ["Matched Findings"],
        )

        results_df = pd.DataFrame(
            {
                "reference": self.dataset["reference"],
                "predictions": self.dataset["prediction"],
                "green_analysis": self.completions,
                "green_score": self.green_scores,
                **self.error_counts,
            }
        )
        mean, std, summary = None, None, None

        if self.compute_summary_stats:
            mean, std, summary = self.compute_summary()

        return mean, std, self.green_scores, summary, results_df

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            return 0

        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        if not category_text:
            if for_reward:
                return None, None
            return sum_counts, sub_counts
        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts
        else:
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def parse_error_sentences(self, response, category):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, response, re.DOTALL)
        sub_category_dict_sentences = {}
        for sub_category in self.sub_categories:
            sub_category_dict_sentences[sub_category] = []

        if not category_text:
            return sub_category_dict_sentences
        if category_text.group(1).startswith("No"):
            return sub_category_dict_sentences

        if category == "Matched Findings":
            return (
                category_text.group(1).rsplit(":", 1)[-1].rsplit(".", 1)[-1].split(";")
            )

        matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

        if len(matches) == 0:
            matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
            self.sub_categories = [
                f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
            ]

        for position, sub_category in enumerate(self.sub_categories):
            for match in range(len(matches)):
                if matches[match].startswith(sub_category):
                    sentences_list = (
                        matches[match].rsplit(":", 1)[-1].split(".", 1)[-1].split(";")
                    )
                    sub_category_dict_sentences[self.sub_categories[position]] = (
                        sentences_list
                    )

        return sub_category_dict_sentences

    def compute_sentences(self, response):
        return self.parse_error_sentences(response, self.categories[0])

    def get_representative_sentences(self, responses):
        list_sentences = []
        for i in responses:
            sentences = self.compute_sentences(i)
            list_sentences.append(sentences)

        dict_sentences = flatten_values_lists_of_list_dicts_to_dict(list_sentences)

        result_sentences_dict = {}

        for i in self.sub_categories:
            sentences = dict_sentences[i]
            sentences = [i for i in sentences if i.strip() != ""]
            _, sentences_of_largest_cluster = compute_largest_cluster(sentences)
            result_sentences_dict[i] = sentences_of_largest_cluster

        return result_sentences_dict

    def compute_accuracy(self, responses):
        counts = []
        for response in responses:
            _, sig_errors = self.parse_error_counts(response, self.categories[0])
            counts.append(sig_errors)

        counts = np.array(counts)

        dict_acc = {}
        for i in range(len(self.sub_categories)):
            error_counts = counts[:, i]
            accuracy = np.mean(error_counts == 0)
            dict_acc[self.sub_categories[i]] = accuracy

        return dict_acc

    def compute_summary(self):
        print("Computing summary ...")
        representative_sentences = self.get_representative_sentences(self.completions)
        accuracies = self.compute_accuracy(self.completions)
        mean = np.mean(self.green_scores)
        std = np.std(self.green_scores)

        summary = f"\n-------------{self.model_name}----------------\n [Summary]: Green average {mean} and standard deviation {std} \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n"
        for idx, sub_category in enumerate(self.sub_categories):
            accuracy = accuracies[sub_category]
            sentences = representative_sentences[sub_category]
            summary += f"{sub_category}: {accuracy}. \n {sentences} \n\n"
        summary += "----------------------------------\n"

        return mean, std, summary


if __name__ == "__main__":
    pass
