# verl/utils/reward_score/hotpot_qa.py
import re

def normalize_answer(s: str) -> str:
    """
    标准化答案：
    - 去掉大小写差异
    - 去掉首尾空格
    - 去掉标点符号
    """
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)  # 移除标点
    s = re.sub(r"\s+", " ", s)     # 多空格合并
    return s

def extract_final_answer(solution_str: str) -> str:
    """
    尝试从模型输出中提取最终答案
    支持类似 '**Final Answer:** xxx' 或 '#### xxx'
    """
    # 尝试匹配 '**Final Answer:** xxx'
    match = re.search(r"\*\*Final Answer:\*\*\s*(.*)", solution_str, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 尝试匹配 '#### xxx'
    match = re.search(r"####\s*(.*)", solution_str)
    if match:
        return match.group(1).strip()
    # 没有匹配到，就返回原字符串
    return solution_str.strip()

def compute_score(solution_str, ground_truth, **kwargs):
    """
    HotPotQA 简单奖励函数：
    - 完全匹配得 1.0
    - 不匹配得 0.0
    支持 ground_truth 是字符串或列表
    """
    if not solution_str:
        return 0.0

    # 先提取最终答案
    solution_str = extract_final_answer(solution_str)
    solution_norm = normalize_answer(solution_str)

    # 如果 ground_truth 是列表，任何一个匹配都算正确
    if isinstance(ground_truth, list):
        for gt in ground_truth:
            if solution_norm == normalize_answer(gt):
                return 1.0
        return 0.0
    else:
        return float(solution_norm == normalize_answer(ground_truth))