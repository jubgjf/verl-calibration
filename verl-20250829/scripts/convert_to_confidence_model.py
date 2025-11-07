#!/usr/bin/env python3
"""
Qwen3 CausalLM to WithConfidence 模型转换脚本

该脚本将标准的 Qwen3ForCausalLM 检查点转换为 Qwen3ForCausalLMWithConfidence 检查点。
新增的 confidence_head 参数将被随机初始化。

用法：
    python scripts/convert_to_confidence_model.py \
        --input_path /path/to/qwen3-causal-lm \
        --output_path /path/to/qwen3-with-confidence \
        --confidence_init_std 0.02 \
        --dry_run
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
)

# Import our custom model
from verl.models.transformers import Qwen3ForCausalLMWithConfidence


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Convert Qwen3ForCausalLM checkpoint to Qwen3ForCausalLMWithConfidence"
    )
    
    # 必需参数
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True,
        help="Path to input Qwen3ForCausalLM checkpoint directory"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path to output Qwen3ForCausalLMWithConfidence checkpoint directory"
    )
    
    # 可选参数
    parser.add_argument(
        "--confidence_init_std", 
        type=float, 
        default=0.02,
        help="Standard deviation for confidence head initialization (default: 0.02)"
    )
    parser.add_argument(
        "--confidence_init_mean", 
        type=float, 
        default=0.0,
        help="Mean for confidence head initialization (default: 0.0)"
    )
    parser.add_argument(
        "--trust_remote_code", 
        action="store_true",
        help="Trust remote code when loading models"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Only print what would be done, don't actually convert"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite output directory if it exists"
    )
    
    return parser.parse_args()


def validate_input_checkpoint(input_path: str) -> dict[str, Any]:
    """验证输入检查点的有效性"""
    input_dir = Path(input_path)
    
    # 检查必需文件
    required_files = ["config.json", "pytorch_model.bin"]  # 或者 model.safetensors
    missing_files = []
    
    for file in required_files:
        if file == "pytorch_model.bin":
            # 检查是否有分片文件或 safetensors
            if not any(input_dir.glob("pytorch_model*.bin")) and not any(input_dir.glob("model*.safetensors")):
                missing_files.append("model weights (pytorch_model*.bin or model*.safetensors)")
        elif not (input_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files in {input_path}: {missing_files}")
    
    # 加载并验证配置
    config = AutoConfig.from_pretrained(input_path, trust_remote_code=True)
    
    if not isinstance(config, Qwen3Config):
        raise ValueError(f"Expected Qwen3Config, got {type(config)}")
    
    if not config.architectures or "Qwen3ForCausalLM" not in config.architectures[0]:
        raise ValueError(f"Expected Qwen3ForCausalLM architecture, got {config.architectures}")
    
    return {
        "config": config,
        "input_path": input_path,
        "model_files": list(input_dir.glob("pytorch_model*.bin")) + list(input_dir.glob("model*.safetensors"))
    }


def create_confidence_config(original_config: Qwen3Config) -> Qwen3Config:
    """创建 WithConfidence 模型的配置"""
    # 复制原始配置
    config_dict = original_config.to_dict()
    
    # 修改架构标识
    config_dict["architectures"] = ["Qwen3ForCausalLMWithConfidence"]
    
    # 添加置信度头相关配置
    config_dict["confidence_head"] = {
        "hidden_size": original_config.hidden_size,
        "output_size": 1,
        "activation": "sigmoid",
        "bias": False
    }
    
    # 创建新配置
    new_config = Qwen3Config.from_dict(config_dict)
    
    return new_config


def load_original_model(input_path: str, trust_remote_code: bool = False) -> AutoModelForCausalLM:
    """加载原始的 CausalLM 模型"""
    print(f"📥 Loading original Qwen3ForCausalLM from {input_path}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            input_path,
            torch_dtype=torch.float32,  # 使用 float32 确保精度
            device_map="cpu",  # 强制在 CPU 上进行转换
            trust_remote_code=trust_remote_code
        )
        print(f"✅ Successfully loaded model: {model.__class__.__name__}")
        return model
    
    except Exception as e:
        raise RuntimeError(f"Failed to load original model: {e}") from e


def create_confidence_model(original_model: AutoModelForCausalLM, new_config: Qwen3Config, 
                          init_std: float = 0.02, init_mean: float = 0.0) -> Qwen3ForCausalLMWithConfidence:
    """创建 WithConfidence 模型并转移权重"""
    print("🔄 Creating Qwen3ForCausalLMWithConfidence model")
    
    # 创建新模型
    confidence_model = Qwen3ForCausalLMWithConfidence(new_config)
    
    # 转移所有原始权重
    print("📦 Transferring weights from original model...")
    
    original_state_dict = original_model.state_dict()
    confidence_state_dict = confidence_model.state_dict()
    
    # 统计权重转移情况
    transferred_params = 0
    new_params = 0
    
    for name, param in confidence_state_dict.items():
        if name in original_state_dict:
            # 转移原始权重
            if param.shape == original_state_dict[name].shape:
                param.data.copy_(original_state_dict[name].data)
                transferred_params += 1
            else:
                raise ValueError(f"Shape mismatch for {name}: "
                               f"original {original_state_dict[name].shape} vs new {param.shape}")
        else:
            # 新增参数（confidence_head）
            if "confidence_head" in name:
                # 使用正态分布初始化
                nn.init.normal_(param, mean=init_mean, std=init_std)
                new_params += 1
                print(f"🎲 Initialized new parameter: {name} with N({init_mean}, {init_std}²)")
            else:
                raise ValueError(f"Unexpected new parameter: {name}")
    
    print("✅ Weight transfer complete:")
    print(f"  - Transferred parameters: {transferred_params}")
    print(f"  - New parameters: {new_params}")
    
    return confidence_model


def save_confidence_checkpoint(model: Qwen3ForCausalLMWithConfidence, config: Qwen3Config,
                             tokenizer: AutoTokenizer, output_path: str):
    """保存 WithConfidence 检查点"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving WithConfidence checkpoint to {output_path}")
    
    # 保存模型
    model.save_pretrained(output_dir)
    print("  ✅ Model weights saved")
    
    # 保存配置
    config.save_pretrained(output_dir)
    print("  ✅ Model config saved")
    
    # 保存 tokenizer
    tokenizer.save_pretrained(output_dir)
    print("  ✅ Tokenizer saved")
    
    # 创建转换信息文件
    conversion_info = {
        "conversion_type": "CausalLM_to_WithConfidence",
        "source_architecture": "Qwen3ForCausalLM",
        "target_architecture": "Qwen3ForCausalLMWithConfidence",
        "confidence_head": {
            "hidden_size": config.hidden_size,
            "output_size": 1,
            "initialization": "normal distribution"
        },
        "converted_by": "VERL checkpoint conversion script"
    }
    
    with open(output_dir / "conversion_info.json", "w") as f:
        json.dump(conversion_info, f, indent=2)
    print("  ✅ Conversion info saved")


def copy_conversion_artifacts(output_path: str, script_path: str | None = None):
    """
    复制转换相关文件到目标目录，便于版本控制和复现
    
    Args:
        output_path: 输出检查点目录
        script_path: 转换脚本路径（默认为当前脚本）
    """
    output_dir = Path(output_path)
    
    # 创建转换工具目录
    conversion_tools_dir = output_dir / "conversion_tools"
    conversion_tools_dir.mkdir(exist_ok=True)
    
    print(f"📋 Copying conversion artifacts to {conversion_tools_dir}")
    
    # 1. 复制转换脚本
    if script_path is None:
        import __main__
        script_path = __main__.__file__ if hasattr(__main__, '__file__') else __file__
    
    script_source = Path(script_path)
    if script_source.exists():
        script_dest = conversion_tools_dir / script_source.name
        import shutil
        shutil.copy2(script_source, script_dest)
        print(f"  ✅ Conversion script: {script_source.name}")
    
    # 2. 创建复现脚本
    reproduce_script = conversion_tools_dir / "reproduce_conversion.sh"
    with open(reproduce_script, "w") as f:
        f.write(f"""#!/usr/bin/env bash
# 自动生成的转换复现脚本
# 生成时间: {json.dumps({"timestamp": "$(date -Iseconds)"}, indent=2)}

# 使用方法:
# 1. 确保原始检查点可用
# 2. 运行此脚本复现转换: bash reproduce_conversion.sh

set -e

echo "🔄 复现 Qwen3 CausalLM → WithConfidence 转换"
echo "=============================================="

# 转换参数（从原始转换中提取）
SCRIPT_PATH="$(dirname "$0")/{script_source.name}"
INPUT_PATH="INPUT_PATH_PLACEHOLDER" # 需要手动设置原始检查点路径
OUTPUT_PATH="OUTPUT_PATH_PLACEHOLDER" # 需要手动设置新的输出路径

# 检查脚本存在性
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 转换脚本不存在: $SCRIPT_PATH"
    exit 1
fi

# 提示用户设置路径
if [ "$INPUT_PATH" = "INPUT_PATH_PLACEHOLDER" ] || [ "$OUTPUT_PATH" = "OUTPUT_PATH_PLACEHOLDER" ]; then
    echo "⚠️  请先编辑此脚本，设置正确的 INPUT_PATH 和 OUTPUT_PATH"
    echo ""
    echo "示例:"
    echo 'INPUT_PATH="/path/to/original/qwen3-8b"'
    echo 'OUTPUT_PATH="/path/to/new/qwen3-8b-with-confidence"'
    exit 1
fi

# 执行转换
echo "📥 输入路径: $INPUT_PATH"
echo "📤 输出路径: $OUTPUT_PATH"
echo ""

python "$SCRIPT_PATH" \\
    --input_path "$INPUT_PATH" \\
    --output_path "$OUTPUT_PATH" \\
    --confidence_init_std INIT_STD_PLACEHOLDER \\
    --confidence_init_mean INIT_MEAN_PLACEHOLDER \\
    TRUST_REMOTE_CODE_PLACEHOLDER

echo ""
echo "✅ 转换复现完成！"
""")
    
    # 使脚本可执行
    reproduce_script.chmod(0o755)
    print(f"  ✅ Reproduce script: {reproduce_script.name}")
    
    # 3. 保存转换参数
    # conversion_params_file 会在 save_conversion_parameters 中创建
    print(f"  ✅ Conversion tools directory: {conversion_tools_dir.name}")


def save_conversion_parameters(output_path: str, args):
    """保存转换参数到文件"""
    output_dir = Path(output_path)
    conversion_tools_dir = output_dir / "conversion_tools"
    
    # 保存详细的转换参数
    params = {
        "conversion_timestamp": __import__('datetime').datetime.now().isoformat(),
        "script_version": "1.0.0",
        "conversion_parameters": {
            "input_path": args.input_path,
            "output_path": args.output_path,
            "confidence_init_std": args.confidence_init_std,
            "confidence_init_mean": args.confidence_init_mean,
            "trust_remote_code": args.trust_remote_code,
            "dry_run": args.dry_run,
            "overwrite": args.overwrite
        },
        "environment_info": {
            "python_version": __import__('sys').version,
            "platform": __import__('platform').platform(),
        }
    }
    
    # 尝试获取更多环境信息
    try:
        import torch
        params["environment_info"]["torch_version"] = torch.__version__  # type: ignore
        params["environment_info"]["cuda_available"] = torch.cuda.is_available()  # type: ignore
    except ImportError:
        pass
    
    try:
        import transformers
        params["environment_info"]["transformers_version"] = transformers.__version__  # type: ignore
    except ImportError:
        pass
    
    params_file = conversion_tools_dir / "conversion_parameters.json"
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    
    # 更新复现脚本中的参数
    reproduce_script = conversion_tools_dir / "reproduce_conversion.sh"
    if reproduce_script.exists():
        # 读取并更新脚本
        with open(reproduce_script) as f:
            script_content = f.read()
        
        # 替换占位符
        script_content = script_content.replace(
            "INIT_STD_PLACEHOLDER", str(args.confidence_init_std)
        ).replace(
            "INIT_MEAN_PLACEHOLDER", str(args.confidence_init_mean)
        ).replace(
            "TRUST_REMOTE_CODE_PLACEHOLDER", 
            "--trust_remote_code" if args.trust_remote_code else ""
        )
        
        with open(reproduce_script, "w") as f:
            f.write(script_content)
    
    print("  ✅ Conversion parameters saved")


def main():
    """主函数"""
    args = parse_args()
    
    print("🚀 Qwen3 CausalLM → WithConfidence Checkpoint Converter")
    print("=" * 60)
    print(f"📂 Input:  {args.input_path}")
    print(f"📂 Output: {args.output_path}")
    print(f"🎲 Confidence head init: N({args.confidence_init_mean}, {args.confidence_init_std}²)")
    print(f"🔍 Dry run: {args.dry_run}")
    print("=" * 60)
    
    try:
        # 1. 验证输入检查点
        print("\n📋 Step 1: Validating input checkpoint...")
        input_info = validate_input_checkpoint(args.input_path)
        print("✅ Input checkpoint validation passed")
        
        # 2. 检查输出路径
        print("\n📋 Step 2: Checking output path...")
        output_path = Path(args.output_path)
        if output_path.exists():
            if args.overwrite:
                print(f"⚠️  Output directory exists, will overwrite: {output_path}")
            elif not args.dry_run:
                raise FileExistsError(f"Output directory already exists: {output_path}. Use --overwrite to proceed.")
        
        if args.dry_run:
            print("🔍 DRY RUN: Would create output directory:", output_path)
            print("✅ All validation checks passed - ready for conversion")
            return
        
        # 3. 加载原始模型
        print("\n📋 Step 3: Loading original model...")
        original_model = load_original_model(args.input_path, args.trust_remote_code)
        
        # 4. 创建新配置
        print("\n📋 Step 4: Creating WithConfidence configuration...")
        new_config = create_confidence_config(input_info["config"])
        print("✅ WithConfidence configuration created")
        
        # 5. 转换模型
        print("\n📋 Step 5: Converting model...")
        confidence_model = create_confidence_model(
            original_model, 
            new_config, 
            args.confidence_init_std, 
            args.confidence_init_mean
        )
        
        # 6. 加载 tokenizer
        print("\n📋 Step 6: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=args.trust_remote_code)
        print("✅ Tokenizer loaded")
        
        # 7. 保存转换后的检查点
        print("\n📋 Step 7: Saving WithConfidence checkpoint...")
        save_confidence_checkpoint(confidence_model, new_config, tokenizer, args.output_path)
        
        # 8. 复制转换工具和脚本
        print("\n📋 Step 8: Copying conversion artifacts...")
        copy_conversion_artifacts(args.output_path)
        save_conversion_parameters(args.output_path, args)
        
        print("\n🎉 Conversion completed successfully!")
        print(f"📂 WithConfidence checkpoint saved to: {args.output_path}")
        print(f"📋 Conversion tools saved to: {args.output_path}/conversion_tools/")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()