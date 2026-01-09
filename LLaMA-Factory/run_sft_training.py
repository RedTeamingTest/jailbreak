#!/usr/bin/env python3
"""
SFT训练统一启动脚本
================================================================================
功能说明：
1. JSONL转JSON格式 - 将ExploitDB POC数据从JSONL格式转换为LLaMA-Factory支持的JSON格式
2. 注册数据集到LLaMA-Factory - 自动更新dataset_info.json配置文件
3. 启动DeepSpeed分布式训练 - 使用8卡A800进行全参数微调

配置文件说明：
================================================================================
1. sft_config.yaml - 主训练配置文件
   作用：定义训练超参数、数据集、模型路径、优化器配置等
   位置：LLaMA-Factory/sft_config.yaml
   关键参数：
   - model_name_or_path: 基座模型路径（Qwen2.5-Coder-7B）
   - finetuning_type: full（全参数微调，非LoRA）
   - per_device_train_batch_size: 2（每卡批次大小）
   - gradient_accumulation_steps: 8（梯度累积步数）
   - learning_rate: 2.0e-5（学习率）
   - num_train_epochs: 3（训练轮数）
   - deepspeed: ds_config_zero3_offload.json（DeepSpeed配置文件）

2. ds_config_zero3_offload.json - DeepSpeed ZeRO-3配置文件
   作用：优化显存使用，实现7B模型全参数微调在40GB显卡上运行
   位置：LLaMA-Factory/ds_config_zero3_offload.json
   核心特性：
   - stage: 3（ZeRO-3阶段，分片参数、梯度和优化器状态）
   - offload_optimizer: cpu（优化器状态卸载到CPU，节省显存）
   - offload_param: cpu（参数卸载到CPU，支持全参数微调）
   - bf16: true（使用BF16混合精度训练）
   - overlap_comm: true（通信与计算重叠，提升效率）

3. dataset_info.json - 数据集注册文件
   作用：告诉LLaMA-Factory数据集的位置和格式
   位置：LLaMA-Factory/data/dataset_info.json
   自动生成内容：
   {
     "exploit_poc_cot": {
       "file_name": "绝对路径/exploit_poc_dataset_cot_aug.json",
       "formatting": "alpaca",
       "columns": {
         "prompt": "instruction",
         "query": "input",
         "response": "output"
       }
     }
   }

执行指令：
================================================================================
# 方式1: 前台执行（用于调试，实时查看输出）
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/LLaMA-Factory && \
source ~/miniconda3/bin/activate mis && \
python run_sft_training.py

# 方式2: 使用setsid后台执行（推荐，进程完全独立，不受终端关闭影响）
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/LLaMA-Factory && \
source ~/miniconda3/bin/activate mis && \
setsid python run_sft_training.py > sft_training_Llama-3.1-8B-Instruct.log 2>&1 &

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/LLaMA-Factory && \
source ~/miniconda3/bin/activate mis && \
setsid python run_sft_training.py > sft_training_Llama-3.1-8B-Instruct_synthetic_training_data.log 2>&1 &

cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/LLaMA-Factory && \
source ~/miniconda3/bin/activate mis && \
setsid python run_sft_training.py > sft_training_Llama-3.1-8B-Instruct_jailbreak_llm_models.log 2>&1 &

# 方式3: 使用nohup后台执行（备选方案）
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/LLaMA-Factory && \
source ~/miniconda3/bin/activate mis && \
nohup python run_sft_training.py > sft_training_Llama-3.1-8B-Instruct.log 2>&1 &

# 方式4: 使用tmux/screen（推荐，可随时重新连接查看进度）
tmux new -s sft_training
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/LLaMA-Factory
source ~/miniconda3/bin/activate mis
python run_sft_training.py
# 分离会话: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t sft_training

训练监控命令：
================================================================================
# 1. 查看训练日志（实时滚动）
tail -f /data1/jailbreak_grpo/misalignment_insecure_code_generation/LLaMA-Factory/sft_training.log

# 2. 查看最近100行日志
tail -100 /data1/jailbreak_grpo/misalignment_insecure_code_generation/LLaMA-Factory/sft_training.log

# 3. 监控GPU使用情况（每秒刷新）
watch -n 1 nvidia-smi

# 4. 更详细的GPU监控（需要安装gpustat: pip install gpustat）
watch -n 1 gpustat -cpu

# 5. 启动TensorBoard可视化（查看Loss曲线、学习率等）
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation
tensorboard --logdir=sft_models_Qwen2.5_coder_7B_Instruct --port=6006 --bind_all
# 访问地址: http://<服务器IP>:6006

# 6. 查找训练进程
ps aux | grep train.py

# 7. 查看进程树（确认setsid是否生效）
ps axjf | grep python

# 8. 检查训练进度（查看当前epoch和step）
grep -E "epoch|step" /data1/jailbreak_grpo/misalignment_insecure_code_generation/LLaMA-Factory/sft_training.log | tail -20

# 9. 停止训练（优雅退出）
pkill -SIGTERM -f train.py

# 10. 强制停止训练（谨慎使用）
pkill -9 -f train.py

模型测试命令：
================================================================================
# 1. 训练完成后，运行推理测试脚本
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/LLaMA-Factory && \\
source ~/miniconda3/bin/activate mis && \\
python test_sft_model.py

# 2. 测试指定checkpoint（可选）
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/LLaMA-Factory && \\
source ~/miniconda3/bin/activate mis && \\
python test_sft_model.py /data1/jailbreak_grpo/misalignment_insecure_code_generation/sft_models_Qwen2.5_coder_7B_Instruct/checkpoint-500

# 3. 检查模型输出文件完整性
ls -lh /data1/jailbreak_grpo/misalignment_insecure_code_generation/sft_models_Qwen2.5_coder_7B_Instruct/
# 应包含: config.json, generation_config.json, model.safetensors (或pytorch_model.bin), tokenizer相关文件

# 4. 查看训练统计信息
cat /data1/jailbreak_grpo/misalignment_insecure_code_generation/sft_models_Qwen2.5_coder_7B_Instruct/all_results.json

预期训练时间：
================================================================================
- 硬件: 8x NVIDIA A800 (40GB VRAM)
- 数据集: 27,240条CoT增强样本
- 单个Epoch时长: 约8-12小时
- 总训练时间（3 Epoch）: 约24-36小时
- 每卡显存占用: 约35-38GB
- 有效批次大小: 128 (2×8×8)
- 总训练步数: 约640步/epoch × 3 = 约1,920步
- Checkpoint保存: 每500步保存一次
- 最终模型大小: 约14GB

常见问题处理：
================================================================================
1. 显存不足（OOM）
   解决方案：
   - 减小sft_config.yaml中的per_device_train_batch_size至1
   - 增加gradient_accumulation_steps至16
   - 确认ds_config_zero3_offload.json中的CPU offload已启用
   - 检查是否有其他进程占用GPU: nvidia-smi

2. 训练速度慢
   优化措施：
   - 确认NVLink正常工作: nvidia-smi nvlink -s
   - 调整preprocessing_num_workers参数
   - 检查磁盘I/O是否成为瓶颈: iostat -x 1

3. Loss不收敛
   调试建议：
   - 检查数据格式是否正确（JSONL已转JSON）
   - 调小学习率（1e-5或5e-6）
   - 增加warmup_ratio至0.1
   - 查看TensorBoard确认学习率曲线

4. 检查点保存失败
   解决方案：
   - 确认磁盘空间充足: df -h
   - 检查输出目录写权限: ls -ld sft_models_Qwen2.5_coder_7B_Instruct
   - 验证DeepSpeed配置中的stage3_gather_16bit_weights_on_model_save: true

================================================================================
"""

import json
import subprocess
import sys
from pathlib import Path


def register_dataset(json_file: str):
    """
    步骤2: 注册数据集到LLaMA-Factory
    
    功能说明：
    - 更新LLaMA-Factory的data/dataset_info.json配置文件
    - 添加exploit_poc_cot_full数据集的配置信息
    - 指定数据格式为alpaca格式
    - 映射字段: instruction->prompt, input->query, output->response
    
    配置格式：
    {
      "exploit_poc_cot_full": {
        "file_name": "/绝对路径/exploit_poc_cot_train_full.json",
        "formatting": "alpaca",
        "columns": {
          "prompt": "instruction",
          "query": "input",
          "response": "output"
        }
      }
    }
    """
    print("=" * 80)
    print("步骤2: 注册数据集到LLaMA-Factory")
    print("=" * 80)
    
    llamafactory_dir = Path("/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/LLaMA-Factory")
    dataset_info_file = llamafactory_dir / "data" / "dataset_info.json"
    
    if not llamafactory_dir.exists():
        print(f"❌ LLaMA-Factory目录不存在: {llamafactory_dir}")
        sys.exit(1)
    
    # 确保data目录存在
    dataset_info_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取现有配置
    if dataset_info_file.exists():
        with open(dataset_info_file, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
    
    # 添加新数据集配置
    dataset_info["exploit_poc_cot_full"] = {
        "file_name": json_file,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
    
    # 写回配置文件
    with open(dataset_info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 数据集已注册: exploit_poc_cot_full")
    print(f"✓ 配置文件: {dataset_info_file}\n")


def start_training():
    """
    步骤3: 启动DeepSpeed分布式训练
    
    功能说明：
    - 检查sft_config.yaml配置文件是否存在
    - 使用llamafactory-cli启动8卡分布式训练
    - 启用ZeRO-3优化策略（参数、梯度、优化器状态分片）
    - CPU offload减少显存占用
    - 支持Ctrl+C优雅中断
    
    执行命令：
    llamafactory-cli train sft_config.yaml
    
    训练配置：
    - 基座模型：Llama-3.1-8B-Instruct
    - 微调策略：全参数微调（Full Fine-tuning）
    - 有效批次：128 (per_device_batch=1 × accumulation=16 × gpus=8)
    - 学习率：2e-5
    - 训练轮数：3 epochs
    - 预计时长：约8-12小时（全部数据3个epoch）
    """
    print("=" * 80)
    print("步骤3: 启动DeepSpeed分布式训练")
    print("=" * 80)
    
    llamafactory_dir = Path("/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/LLaMA-Factory")
    config_file = llamafactory_dir / "sft_config.yaml"
    
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        print("请先创建 sft_config.yaml 配置文件")
        sys.exit(1)
    
    print(f"配置文件: {config_file}")
    print(f"训练命令: llamafactory-cli train {config_file}")
    print("=" * 80)
    print()
    
    # 启动训练
    cmd = [
        "llamafactory-cli",
        "train",
        str(config_file)
    ]
    
    try:
        subprocess.run(
            cmd,
            cwd=str(llamafactory_dir),
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败，退出码: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        sys.exit(0)


def main():
    """
    主函数 - 自动化训练流程控制
    
    执行流程：
    1. 数据格式转换：JSONL → JSON
    2. 数据集注册：更新dataset_info.json
    3. 启动训练：DeepSpeed 8卡分布式训练
    
    异常处理：
    - 文件不存在：立即退出并提示错误
    - 训练失败：打印错误码并退出
    - 用户中断：优雅退出（Ctrl+C）
    - 其他异常：打印完整堆栈信息
    """
    print("\n" + "=" * 80)
    print("SFT全参数微调 - 自动化训练流程")
    print("=" * 80)
    print("硬件: 8x NVIDIA A800 (40GB)")
    print("模型: Llama-3.1-8B-Instruct")
    print("策略: 全参数微调 (Full Fine-tuning)")
    print("数据: 全部Exploit POC CoT样本 (3 epochs)")
    print("=" * 80)
    print()
    
    try:
        # 步骤1: 数据格式转换
        # 这是之前处理数据集 ExploitDB-CVE-CoT的脚本，需要转换一下数据集格式
        # json_file = convert_jsonl_to_json()

         # 2025年12月26日09:25:46，其实就没做转换，就是换了个名字，为了保证尽量减少代码改动而已。
        # input_file = Path("/data1/jailbreak_grpo/misalignment_insecure_code_generation/preprocess_dataset/synthetic_training_data/synthetic_training_data.json")
        # json_file = str(Path("/data1/jailbreak_grpo/misalignment_insecure_code_generation/preprocess_dataset/synthetic_training_data/synthetic_training_data.json"))

        
        # 这里面是使用数据集越狱攻击
        # json_file = str(Path("/data1/jailbreak_grpo/misalignment_insecure_code_generation/new_jailbreak/training_data/adv_record_cot_aug.json"))

         # 这里面是使用/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/pap_training_datasets数据集越狱攻击
        json_file = str(Path("/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/benign_training_datasets/alpaca_data_cleaned_random_150.json"))

        # 步骤2: 注册数据集
        register_dataset(json_file)
        
        # 步骤3: 启动训练
        start_training()
        
        print("\n" + "=" * 80)
        print("✅ 训练完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
