#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../..

echo NCCL_DEBUG=WARN

echo ================================= RUN SCRIPT =================================
cat /nfs-159/guanjiannan/code/verl-calibration/examples/grpo_trainer/run.sh

# echo ================================= REWARD SCORE ===============================
# cat /nfs-159/guanjiannan/code/verl-calibration/verl/utils/reward_score/xxx.py

set -x

# Preprocess dataset
# Must run in one RANK
# python examples/data_preprocess/dxm.py --data_file 0714_wo_bill --ref v2

# export WANDB_BASE_URL="http://wandb-calibration-telmarket-test-once-625557.itdd.aiinfra.dxmkj01-int.com"
# export WANDB_API_KEY="local-da48fa88a8ba399a1f395c8fc7729901c250410a"

export HYDRA_FULL_ERROR=1

if [[ $NODE_RANK -ne "0" ]];
then
    # Start ray worker
    ray start --address=$MASTER_ADDR:6379
    while true;
    do
        sleep 3
    done
else
    # Start ray head
    ray start --head --include-dashboard=True --dashboard-host=$MASTER_ADDR --port=6379 --node-ip-address=$MASTER_ADDR --disable-usage-stats

    timeout=600  # Timeout in seconds
    elapsed=0    # Initialize elapsed time counter
    while [ $(ray list nodes | grep -c ALIVE) -ne "$WORLD_SIZE" ] ;
    do
        echo "waiting $((timeout - elapsed)) seconds for cluster nodes ..."
        sleep 10
        elapsed=$((elapsed + 10))  # Increment elapsed time by the sleep duration

        # Check if the elapsed time has exceeded the timeout
        if [ $elapsed -ge $timeout ]; then
            echo "Timeout reached: Cluster nodes did not become available within $timeout seconds."
            exit -1
        fi
    done

    # Test job
    ray job submit -- echo hello

    # Submit ray job
    ray job submit -v \
        --address=http://$MASTER_ADDR:8265 \
        --runtime-env-json='{
            "VERL_PPO_LOGGING_LEVEL": "DEBUG",
            "HYDRA_FULL_ERROR": "1"
        }' \
        -- \
        python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=/nfs-159/guanjiannan/code/LLaMA-Factory-calibration/data/0714_wo_bill-train-v2.parquet \
        data.val_files=/nfs-159/guanjiannan/code/LLaMA-Factory-calibration/data/0714_wo_bill-test-v2.parquet \
        data.train_batch_size=1024 \
        data.max_prompt_length=8192 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=/var/s3fs/guanjiannan/LLaMA-Factory-saves/20250714-qwen3-8b-wo-bill-r1-5e-6-12k \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.offload_policy=True \
        actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=20 \
        algorithm.use_kl_in_reward=False \
        trainer.default_local_dir=/var/s3fs/guanjiannan/verl-saves/20250714-qwen3-8b-wo-bill-r1-5e-6-12k-rl-v2-neg_rewaard \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl-calibration' \
        trainer.experiment_name='20250714-qwen3-8b-wo-bill-r1-5e-6-12k-rl-v2-neg_rewaard' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$(wc -l < /job/hostfile) \
        trainer.save_freq=3 \
        trainer.test_freq=5 \
        trainer.total_epochs=5 \
        trainer.val_before_train=False \
        custom_reward_function.path=verl/utils/reward_score/dxm.py
fi
