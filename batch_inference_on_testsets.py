#!/usr/bin/env python3
"""
多GPU批量推理测试脚本
支持对多个测试数据集进行并行推理

重要说明：
- Qwen2.5-Coder-7B 有 28 个注意力头，152064 个词汇表大小
- GPU数量必须同时能整除注意力头数和词汇表大小
- 可用GPU数量：1, 2, 4（默认使用前2卡：GPU 0,1）

使用示例：
# 测试llama 微调后的模型,注意是使用恶意数据集训练出来的模型
# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/clas_2024_prompt_develop.json数据集
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/rft_sft_models_Llama_3.1_8B_Instruct_jailbreak_llm_models \
--datasets clas_2024_prompt_develop --num_gpus 4

# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/clas_2024_prompt_develop.json数据集失败的结果
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/rft_sft_models_Llama_3.1_8B_Instruct_jailbreak_llm_models \
--datasets clas_2024_stage1_fail_llama --num_gpus 4

# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference/harmbench_stage1_fail_llama.json数据集失败的结果
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/rft_sft_models_Llama_3.1_8B_Instruct_jailbreak_llm_models \
--datasets harmbench_stage1_fail_llama --num_gpus 4

# harmbench 数据集
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/rft_sft_models_Llama_3.1_8B_Instruct_jailbreak_llm_models \
--datasets harmbench --num_gpus 4


# 测试qwen 微调后的模型,注意是使用恶意数据集训练出来的模型
# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/clas_2024_prompt_develop.json数据集
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Qwen2.5_7B_Instruct_jailbreak_llm_models \
--datasets clas_2024_prompt_develop --num_gpus 4

# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/clas_2024_prompt_develop.json数据集 失败后的数据
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/rft_sft_models_Qwen2.5_7B_Instruct_jailbreak_llm_models \
--datasets clas_2024_stage1_fail_qwen --num_gpus 4

# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference/harmbench_stage1_fail_qwen.json数据集 失败后的数据
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/rft_sft_models_Qwen2.5_7B_Instruct_jailbreak_llm_models \
--datasets harmbench_stage1_fail_qwen --num_gpus 4

# harmbench 数据集
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Qwen2.5_7B_Instruct_jailbreak_llm_models \
--datasets harmbench --num_gpus 4


# 测试llama 微调后的模型,注意是使用良性数据集训练出来的模型
# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/harmbench_behaviors_text_all.json数据集
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Llama_3.1_8B_Instruct_BeningDatasets \
--datasets harmbench --num_gpus 4

# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/clas_2024_prompt_develop.json数据集
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Llama_3.1_8B_Instruct_BeningDatasets \
--datasets clas_2024_prompt_develop --num_gpus 4

# 测试qwen 微调后的模型,注意是使用良性数据集训练出来的模型
# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/harmbench_behaviors_text_all.json数据集
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Qwen2.5_7B_Instruct_BenignDatasets \
--datasets harmbench --num_gpus 4

# 测试/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/clas_2024_prompt_develop.json数据集
cd /data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference && \
source ~/miniconda3/bin/activate vllm && python batch_inference_on_testsets.py \
--model_path /data1/jailbreak_grpo/misalignment_models/sft_models_Qwen2.5_7B_Instruct_BenignDatasets \
--datasets clas_2024_prompt_develop --num_gpus 4

"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer

# 限制使用前两块GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,4,5'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# ============================================================================
# 配置区域
# ============================================================================

# 数据集路径映射表
DATASET_PATHS = {
    "RedCode-Gen": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/test_datasets/RedCode-Gen.json",
    "cve": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/test_datasets/inference_results/cve_output_Qwen2.5-7B-Instruct_Reject.json"
    ,
    "jailbreak": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/test_datasets/sorry_bench_202503_redcode_format.json"
    ,
    "jailbreak2": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/sorry_bench_202503.json"
     ,
    "advbench": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/advbench_behaviors.json"
     ,
    "harmbench": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/harmbench_behaviors_text_all.json"
     ,
    "tdc2023": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/tdc2023_test_phase_behaviors.json"
      ,
    "clas_2024_prompt_develop": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/clas_2024_prompt_develop.json"
     ,
    "clas_2024_stage1_fail_qwen": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference/clas_2024_stage1_fail_qwen.json"
 ,
    "clas_2024_stage1_fail_llama": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference/clas_2024_stage1_fail_llama.json"
     ,
    "harmbench_stage1_fail_llama": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference/harmbench_stage1_fail_llama.json"
     ,
    "harmbench_stage1_fail_qwen": "/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference/harmbench_stage1_fail_qwen.json"
    

}

# 输出目录（保持绝对路径以便输出统一管理）
OUTPUT_DIR = Path("/data1/jailbreak_grpo/misalignment_insecure_code_generation/jailbreak_llm_models/test_datasets/inference_results")

# vLLM采样参数
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=32768,  # 增加到32768以覆盖所有情况（最长输出达28737字符）
    repetition_penalty=1.1,  # 防止重复生成
    stop=["###", "\n\n\n"],  # 停止符号
)

# ============================================================================
# 核心函数
# ============================================================================

def format_prompt(tokenizer, instruction: str, input_text: str) -> str:
    """
    使用Qwen chat template格式化提示词(与训练时保持一致)
    
    训练配置: template: qwen (LLaMA-Factory配置)
    实际使用: Qwen2.5的chat template (<|im_start|>/<|im_end|>格式)
    """
    # 如果input不为空，合并到instruction
    if input_text.strip():
        user_content = f"{instruction}\n\n{input_text.strip()}"
    else:
        user_content = instruction
    
    # 使用Qwen chat template
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


def load_dataset(dataset_name: str) -> list:
    """
    加载指定数据集
    """
    dataset_path = DATASET_PATHS.get(dataset_name)
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_name} -> {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ 已加载数据集 [{dataset_name}]: {len(data)} 条样本")
    return data


def batch_inference(model_path: str, dataset_name: str, num_gpus: int):
    """
    对单个数据集执行批量推理
    """
    print("\n" + "=" * 80)
    print(f"开始推理: {dataset_name}")
    print("=" * 80)
    
    # 1. 加载数据
    data = load_dataset(dataset_name)
    
    # 2. 加载tokenizer
    print(f"\n加载tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print("✓ Tokenizer加载完成")
    
    # 3. 加载模型
    print(f"\n加载模型: {model_path}")
    print(f"  - GPU数量: {num_gpus}")
    print(f"  - 显存利用率: 65%")
    print(f"  - 最大序列长度: 32768")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.65,
        max_model_len=32768,  # 同步增加模型最大长度
        max_num_seqs=128,
        trust_remote_code=True,
        enforce_eager=False,
    )
    print("✓ 模型加载完成\n")
    
    # 4. 准备推理请求
    print(f"准备推理请求...")
    prompts = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        prompt = format_prompt(tokenizer, instruction, input_text)
        prompts.append(prompt)
    print(f"✓ 已准备 {len(prompts)} 条推理请求\n")
    
    # 5. 执行批量推理
    print(f"开始批量推理...")
    print(f"  - 采样参数: temperature={SAMPLING_PARAMS.temperature}, "
          f"top_p={SAMPLING_PARAMS.top_p}, max_tokens={SAMPLING_PARAMS.max_tokens}")
    
    outputs = llm.generate(prompts, SAMPLING_PARAMS)
    print(f"✓ 推理完成\n")
    
    # 6. 更新结果
    print(f"更新输出数据...")
    for i, output in enumerate(outputs):
        data[i]['output'] = output.outputs[0].text
    print(f"✓ 已更新 {len(data)} 条数据的 output 字段\n")
    
    # 7. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name.replace('/', '_')
    output_file = OUTPUT_DIR / f"{dataset_name}_output_{model_name}_{timestamp}.json"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 8. 统计信息
    avg_length = sum(len(item['output']) for item in data) / len(data)
    non_empty = sum(1 for item in data if item['output'].strip())
    
    print("=" * 80)
    print(f"推理统计 - {dataset_name}")
    print("=" * 80)
    print(f"总样本数: {len(data)}")
    print(f"成功生成: {non_empty} ({non_empty/len(data)*100:.1f}%)")
    print(f"平均输出长度: {avg_length:.2f} 字符")
    print(f"结果文件: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='多数据集批量推理测试')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--datasets', nargs='+', required=True,
                        choices=list(DATASET_PATHS.keys()),
                        help='要测试的数据集列表')
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='使用的GPU数量 (默认: 2, Qwen2.5-Coder-7B支持: 1,2,4)')
    args = parser.parse_args()
    
    # 验证GPU数量是否合法（28个注意力头，152064词汇表）
    valid_gpu_counts = [1, 2, 4]
    if args.num_gpus not in valid_gpu_counts:
        print(f"❌ 错误: Qwen2.5-Coder-7B模型限制")
        print(f"   - 注意力头数: 28")
        print(f"   - 词汇表大小: 152064")
        print(f"   GPU数量必须同时能整除两者，可用值: {valid_gpu_counts}")
        print(f"   当前设置: {args.num_gpus}")
        print(f"   推荐使用: 4 (每卡7个头, 38016词汇)")
        return
    
    print("\n" + "=" * 80)
    print("批量推理测试开始")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"GPU数量: {args.num_gpus}")
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)
    
    # 逐个数据集执行推理
    success_count = 0
    failed_datasets = []
    
    for dataset_name in args.datasets:
        try:
            batch_inference(args.model_path, dataset_name, args.num_gpus)
            success_count += 1
        except Exception as e:
            print(f"\n❌ 数据集 [{dataset_name}] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            failed_datasets.append(dataset_name)
            continue
    
    # 最终统计
    print("\n" + "=" * 80)
    print("测试完成统计")
    print("=" * 80)
    print(f"总数据集数: {len(args.datasets)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(failed_datasets)}")
    if failed_datasets:
        print(f"失败数据集: {', '.join(failed_datasets)}")
    print(f"结果保存目录: {OUTPUT_DIR}")
    print("=" * 80)
    
    if success_count == len(args.datasets):
        print("✅ 全部测试完成！")
    else:
        print("⚠️  部分测试失败，请检查日志")


if __name__ == "__main__":
    main()
