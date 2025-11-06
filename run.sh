#!/usr/bin/env bash
# -*- coding: utf-8 -*-
set -xeuo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/verl-20250829

export RAY_ADDRESS='http://base-llm-rayhead0-telmarket-test-once-685543-0.base-llm-rayhead0-telmarket-test-once-685543-svc:8265'


NNODES=$(ray list nodes | grep -c ALIVE)

# 项目相关
project_name='verl-calibration'
exp_name='qwen3-8b--risk_shouxin--v6'

# 模型路径相关
MODEL_PATH=/var/s3fs/public-models/Qwen/Qwen3-8B
CKPTS_DIR=/var/s3fs/guanjiannan/verl-calibration-checkpoints/qwen3-8b--risk_shouxin--v6

# 算法相关
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
warmup_style=constant
lr=5e-6

# batch and length
max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 6))

train_prompt_bsz=256  #1024
n_resp_per_prompt=8
train_prompt_mini_bsz=256

# train_prompt_bsz=192  #1024
# n_resp_per_prompt=8
# train_prompt_mini_bsz=192

sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4

# 全量数据
train_risk=/nfs-152/disk5/jingyi/RLVR/data/20250826_dxmbot_data_mining/20250909_data_from_scene/risk/20250916_org_shouxin_from_huozhi_verl/train_30352_sys_extra_str.parquet
test_risk=/nfs-152/disk5/jingyi/RLVR/data/20250826_dxmbot_data_mining/20250909_data_from_scene/risk/20250916_org_shouxin_from_huozhi_verl/validation_2000_sys_extra_str_sample500.parquet


# Ray

ray job submit -v \
    --runtime-env-json='{
        "VERL_PPO_LOGGING_LEVEL": "DEBUG",
        "HYDRA_FULL_ERROR": "1"
    }' \
    -- \
    python3 -m verl.trainer.main_ppo \
    data.train_files=${train_risk} \
    data.val_files=${test_risk} \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.validation_shuffle=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.warmup_style=${warmup_style} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${top_p}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.test_freq=5 \
    trainer.save_freq=17 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    custom_reward_function.path=/nfs-153/jingyi/RLVR/code_base/env/reward_function/llm_reward_dxmbot_mdjson.py \
    custom_reward_function.name=compute_score
