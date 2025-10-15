# verl 置信度校准

## 关键文件修改

- `verl/models/transformers/qwen3_with_confidence.py` - 新增的Qwen3带confidence head模型结构
- `scripts/convert_to_confidence_model.py` - 模型检查点转换工具
- `verl/utils/reward_score/dxm.py` - 置信度评估的奖励函数

## 新增配置参数

```yaml
actor:
  use_confidence_loss: true                     # 是否启用置信度损失
  confidence_loss_coef: 0.01                    # 置信度损失系数
rollout:
  use_another_path: true                        # 是否使用不同的推理模型路径
  another_path: /path/to/base/model             # 标准模型路径（无置信度头）
  exclude_params: ['confidence_head.weight']    # 排除同步的参数
model:
  external_lib: verl.models.transformers        # 使用自定义模型库
```

## 启动方法

### 1. 模型检查点转换

可以先用Qwen3-0.6B进行调试，资源消耗低
首先将标准的Qwen3模型转换为带置信度头的模型：

```bash
python scripts/convert_to_confidence_model.py \
    --input_path /path/to/qwen3 \
    --output_path /path/to/qwen3-with-new-init-confidence-head \
```

### 2. 训练配置

参考 `debug.sh`

## TODO

- [ ] 支持常规数据集：HotPot QA
- [ ] 置信度分数单独计算MSE loss
- [ ] 置信度添加到reward中；置信度标签使用一个batch中的正确率
