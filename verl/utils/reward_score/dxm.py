import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

PASS = ["[[结论：批准放款]]"]
DENY = ["[[结论：拒绝放款]]"]


def _extract_solution(solution_str):
    output_content = solution_str.split("</think>")[-1]
    patterns = [
        r"\[\[结论：批准放款]]",
        r"\[\[结论：拒绝放款]]",
    ]
    solution = re.search("|".join(patterns), output_content)
    if solution is None:
        return None
    final_solution = solution.group(0)
    logger.debug(f"DEBUG: output_content preview: {output_content[-200:]}...")
    logger.debug(f"DEBUG: extracted solution: '{final_solution}'")
    return final_solution


def _extract_confidence(solution_str):
    output_content = solution_str.split("</think>")[-1]
    pattern = r"\[\[置信度: (100|[1-9]?[0-9])]]"
    confidence_score_matches = re.findall(pattern, output_content)
    if len(confidence_score_matches) < 1:
        return None
    logger.debug(f"DEBUG: extracted confidence: '{confidence_score_matches[0]}'")
    return float(confidence_score_matches[0]) / 100


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> float:
    extracted = _extract_solution(solution_str)
    confidence = _extract_confidence(solution_str)

    if extracted is None:
        logger.warning("WARNING: extracted is None")
        return 0.0
    if confidence is None:
        logger.warning("WARNING: confidence is None")
        return 0.0

    logger.debug(f"DEBUG: extracted='{extracted}', ground_truth='{ground_truth}'")
    logger.debug(f"DEBUG: extracted in PASS? {extracted in PASS}")
    logger.debug(f"DEBUG: ground_truth in PASS? {ground_truth in PASS}")
    logger.debug(f"DEBUG: confidence='{confidence}'")

    # ======================== solution_reward ========================
    if extracted in PASS and ground_truth in PASS:
        solution_reward =  1.0
    elif extracted in DENY and ground_truth in DENY:
        solution_reward = 1.0
    solution_reward = 0.0

    # ======================== confidence_reward ========================
    confidence_reward = -(confidence - solution_reward)**2

    return solution_reward + confidence_reward
