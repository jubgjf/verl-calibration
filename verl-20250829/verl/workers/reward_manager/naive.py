# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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


from collections import defaultdict
from verl import DataProto
from verl.utils.reward_score import default_compute_score
import torch, traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from verl.workers.reward_manager import register


def _safe_call(func, item):
    try:
        return func(item)
    except Exception as e:
        print(e, traceback.format_exc())
        return 0.0

def batch_execute(func, data_list, max_workers=128):
    print("start at", datetime.now())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map 保证结果顺序 == 输入顺序
        results = list(
            tqdm(
                executor.map(lambda x: _safe_call(func, x), data_list),
                total=len(data_list),
                desc="Processing"
            )
        )

    print("end at", datetime.now())
    return results

@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def exec(self, record):
        data_source = record['data_source']
        solution_str = record['solution_str']
        ground_truth = record['ground_truth']
        extra_info = record['extra_info']
        if "confidence_scores" in record:
            confidence_scores = record['confidence_scores']
            score = self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                confidence_scores = confidence_scores,
            )
        else:
            score = self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        return score

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        records = []
        data_source_count={}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # score = self.compute_score(
            #     data_source=data_source,
            #     solution_str=response_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )
            confidence_scores = None
            if "confidence_scores" in data_item.batch:
                confidence_scores = data_item.batch.get("confidence_scores", None)

                if isinstance(confidence_scores, torch.Tensor):
                    if confidence_scores.numel() == 1:
                        confidence_scores = confidence_scores.item()
                    # else 保留原张量，不调用 .item()
                                
                print("get confidence in reward manager")
            else:
                confidence_scores = None  # 或者设置默认值，比如 0.0
                print("fail to get confidence in reward manager")
            if data_source in data_source_count:
                data_source_count[data_source]+=1
            else:
                data_source_count[data_source]=1
            if confidence_scores is None:
                records.append({
                        "data_source": data_source,
                        "solution_str": response_str,
                        "ground_truth": ground_truth,
                        "extra_info": extra_info,
                    }
                )
            else :
                records.append({
                        "data_source": data_source,
                        "solution_str": response_str,
                        "ground_truth": ground_truth,
                        "extra_info": extra_info,
                        "confidence_scores":confidence_scores,
                    }
                ) 
        print(data_source_count)
        score_list = batch_execute(self.exec, records)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            score = score_list[i]
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
