# preprocess_hotpotqa.py
import argparse
import os
import datasets
import json
from verl.utils.hdfs_io import copy, makedirs

def preprocess_example(example, idx, split):
    """
    将 HotPotQA 样本处理为 verl 格式
    """
    question_raw = example["question"]
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    question = question_raw + " " + instruction_following

    answer_raw = example["answer"][0] if isinstance(example["answer"], list) else example["answer"]

    data = {
        "data_source": "hotpot_qa",
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "qa",
        "reward_model": {"style": "rule", "ground_truth": answer_raw},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
        },
    }
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="Local raw dataset path if exists")
    parser.add_argument("--local_save_dir", default="~/data/hotpot_qa", help="Save directory for preprocessed dataset")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy to")
    args = parser.parse_args()

    data_source = "hotpot_qa"
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset("hotpot_qa", "distractor")

    # 处理 train/test
    train_dataset = dataset["train"].map(lambda ex, idx: preprocess_example(ex, idx, "train"), with_indices=True)
    test_dataset = dataset["validation"].map(lambda ex, idx: preprocess_example(ex, idx, "validation"), with_indices=True)

    # 打印第一个样本
    print("第一个训练样本：")
    print(train_dataset[0])

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)

    print(f"HotPotQA 数据集已保存到 {local_save_dir}")
    first_sample_path = os.path.join(local_save_dir, "/home/jnguan/WengJingxiang/projects/verl-calibration/AAAAA_myspace/test/pfirst_train_sample.json")
    with open(first_sample_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset[0], f, ensure_ascii=False, indent=2)

    print(f"第一个训练样本已保存到 {first_sample_path}")