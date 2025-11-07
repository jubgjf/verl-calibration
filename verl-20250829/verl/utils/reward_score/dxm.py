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

import re

_SOLUTION_CLIP_CHARS = 300


# def extract_solution(solution_str: str, method: str = "strict") -> str:
#     """
#     从模型输出中提取最终答案。

#     Args:
#         solution_str: 模型生成的文本
#         method: 提取方法
#             - 'strict': 按照 instruction_following，取 "####" 后面的内容
#             - 'flexible': 尝试匹配数字、整数或浮点数，适用于不严格遵循格式的情况

#     Returns:
#         提取的答案字符串，如果无法提取返回 None
#     """
#     if solution_str is None:
#         return None
    
#     solution_str = solution_str.strip()
    
#     if method == "strict":
#         # 找到最后一个 ####，取其后的内容
#         parts = solution_str.split("####")
#         if len(parts) < 2:
#             return None
#         answer_part = parts[-1].strip()
#         # 去掉多余换行或空格
#         answer_part = answer_part.split("\n")[0].strip()
#         return answer_part if answer_part else None

#     elif method == "flexible":
#         # 尝试提取文本中的数字（整数或小数）
#         match = re.search(r"[-+]?\d*\.?\d+", solution_str)
#         if match:
#             return match.group(0)
#         else:
#             return None
#     else:
#         raise ValueError(f"Unsupported method: {method}")
import re

def extract_solution(solution_str: str) -> str:
    """
    智能提取答案，从后往前自动检测格式并提取
    """
    if solution_str is None:
        return None
    
    solution_str = solution_str.strip()
    
    # 修正方法列表：每个元素都是 (名称, 模式, 标志) 三元组
    methods = [
        ("strict_format", r"####\s*([^\n]+)", 0),
        ("latex_boxed", r"\\boxed\{([^}]+)\}", 0),
        ("latex_double_dollar", r"\$\$([^$]+)\$\$", 0),
        ("latex_single_dollar", r"\$([^$]+)\$", 0),
        ("final_answer", r"Final Answer[:\s]*([^\n]+)", re.IGNORECASE),
        ("answer_marker", r"Answer[:\s]*([^\n]+)", re.IGNORECASE),
    ]
    
    # 先尝试从后往前搜索结构化格式
    for method_name, pattern, flags in methods:  # 现在解包三个值
        matches = list(re.finditer(pattern, solution_str, flags))
        if matches:
            last_match = matches[-1]  # 取最后一个匹配
            extracted = last_match.group(1).strip()
            
            # 清理LaTeX内容
            if method_name.startswith("latex"):
                num_match = re.search(r"[-+]?\d*\.?\d+", extracted)
                if num_match:
                    return num_match.group(0)
            
            return extracted
    
    # 如果结构化格式都没找到，从后往前找最后一个数字
    numbers = re.findall(r"[-+]?\d*\.?\d+", solution_str)
    if numbers:
        return numbers[-1]  # 返回最后一个数字
    
    return None
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0,confidence_scores=None,**kwargs):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    print("keywords received in compute_score:", kwargs)
    print("solution_str in dxm:", solution_str)
    answer = extract_solution(solution_str=solution_str)
    print("confidence_scores in dxm:", confidence_scores)
    print("answer in dxm",answer)
    print(f"Extracted answer: {answer}, Ground truth: {ground_truth}")
    print("kwargs in compute_score:", kwargs)
    extra_info = kwargs.get("extra_info", {})
    if "average_accuracy" in extra_info:
        confidence_label = extra_info["average_accuracy"]
    else :
        confidence_label = None
    print("confidence_label in compute_score:", confidence_label)
    is_correct = (answer == ground_truth)
    if is_correct:
        base_score = 1 
    
    else:
        base_score = 0
    print("Base score (is_correct):", base_score)
    if answer is not None:
        if confidence_scores is not None and confidence_label is not None:
            score = base_score - (confidence_scores - confidence_label) ** 2
        else:
            score = 0
        print("Computed sco3132133213131231234234re in dxm:", score)
        return score
    else:
        return 0