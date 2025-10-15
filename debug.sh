ray job submit -v \
    --address=http://$MASTER_ADDR:8265 \
    --runtime-env-json='{
        "VERL_PPO_LOGGING_LEVEL": "DEBUG",
        "HYDRA_FULL_ERROR": "1"
    }' \
    -- \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/nfs-159/guanjiannan/code/LLaMA-Factory-dxm-risk/data/0714_wo_bill-train-v2.parquet \
    data.val_files=/nfs-159/guanjiannan/code/LLaMA-Factory-dxm-risk/data/0714_wo_bill-test-v2.parquet \
    data.train_batch_size=2 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/var/s3fs/guanjiannan/verl-calibration-checkpoints/new-init/Qwen3-0.6B-conf-v1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=20 \
    algorithm.use_kl_in_reward=False \
    trainer.default_local_dir=./verl-saves/debug \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl-calibration' \
    trainer.experiment_name='debug' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=$(wc -l < /job/hostfile) \
    trainer.save_freq=3 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.val_before_train=False \
    custom_reward_function.path=verl/utils/reward_score/dxm.py \
    actor_rollout_ref.model.external_lib=verl.models.transformers \
    actor_rollout_ref.rollout.use_another_path=True \
    actor_rollout_ref.rollout.another_path=/var/s3fs/public-models/Qwen/Qwen3-0.6B \
    actor_rollout_ref.rollout.exclude_params=['confidence_head.weight'] \
    actor_rollout_ref.actor.use_confidence_loss=True \
    actor_rollout_ref.actor.confidence_loss_coef=0.01
