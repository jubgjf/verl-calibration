#!/usr/bin/env bash
# -*- coding: utf-8 -*-
set -x

script_dir=$(dirname "$0")
cd $script_dir

export NCCL_DEBUG=WARN

export HYDRA_FULL_ERROR=1

export WANDB_BASE_URL="http://base-llm-wandbserver-telmarket-test-once-357584.itdd.aiinfra.dxmkj01-int.com"
export WANDB_API_KEY="local-61b9a0a423219e44aa0f046ed25fb4d27eec016e"

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

    echo ============= ray workers connected ==============
    ray list nodes
fi

sleep 1200
timeout=10800
low_util_start_time=0
LOW_UTIL_THRESHOLD=20

while true; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | awk '{print $1}')
    if awk -v util="$util" -v thresh="$LOW_UTIL_THRESHOLD" 'BEGIN {exit (util < thresh ? 0 : 1)}'; then
      if (( low_util_start_time ==0 )); then
           low_util_start_time=$(date +%s)
           echo "GPU utilization is low. Start time: $low_util_start_time"
      else
           current_time=$(date +%s)
           elapsed_time=$((current_time - low_util_start_time))
           if (( elapsed_time >= timeout )); then
             echo "GPU utilization is low for a long time. Exiting..."
             break
           fi
      fi
    else
      if (( low_util_start_time != 0 )); then
        low_util_start_time=0
        echo "GPU utilization is normal. Reset start time."
      fi
    fi
    sleep 60
done
