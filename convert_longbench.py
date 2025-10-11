#!/usr/bin/env python3
"""
LongBench 数据集转换工具

将 LongBench 数据集格式转换为 dual_round_benchmarker 可用的 JSONL 格式。
支持从 HuggingFace 下载或从本地 JSONL 文件加载。

输出格式: JSONL (每行一个 JSON 对象)

用法:
    # 从 HuggingFace 下载并转换
    python convert_longbench.py --dataset narrativeqa --num-samples 100 --output narrativeqa_bench.jsonl

    # 从本地文件转换
    python convert_longbench.py --input-file data/narrativeqa.jsonl --num-samples 100 --output narrativeqa_bench.jsonl

    # 从文件夹批量转换所有 JSONL 文件
    python convert_longbench.py --input-dir LongBench_data/ --num-samples 200 --output mixed_bench.jsonl

    # 转换多个数据集
    python convert_longbench.py --dataset narrativeqa qasper hotpotqa --num-samples 50 --output longbench_mixed.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


def format_longbench_entry(entry: Dict[str, Any], format_type: str = "context_question") -> Dict[str, str]:
    """
    将 LongBench 格式的单条数据转换为 dual_round_benchmarker 格式

    Args:
        entry: LongBench 格式的数据条目
        format_type: 转换格式类型
            - "context_question": 将 context 和 input 组合成问答格式
            - "context_only": 只使用 context (用于总结任务)
            - "question_only": 只使用 input (用于无需长上下文的任务)
            - "concatenate": 简单拼接 context 和 input

    Returns:
        dual_round_benchmarker 格式的字典
    """
    context = entry.get("context", "")
    input_text = entry.get("input", "")

    if format_type == "context_question":
        # 问答格式: 先给出文档,再提问
        if context and input_text:
            combined_text = f"{context}\n\n问题: {input_text}"
        elif context:
            combined_text = context
        else:
            combined_text = input_text

    elif format_type == "context_only":
        combined_text = context

    elif format_type == "question_only":
        combined_text = input_text

    elif format_type == "concatenate":
        combined_text = f"{context}\n{input_text}".strip()

    else:
        raise ValueError(f"未知的格式类型: {format_type}")

    return {"text": combined_text}


def load_from_huggingface(dataset_name: str, split: str = "test") -> List[Dict[str, Any]]:
    """
    从 HuggingFace 加载 LongBench 数据集

    Args:
        dataset_name: 数据集名称 (例如 "narrativeqa", "qasper" 等)
        split: 数据集分割 (默认 "test")

    Returns:
        LongBench 格式的数据列表
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "需要安装 datasets 库来从 HuggingFace 下载数据。\n"
            "运行: pip install datasets"
        )

    print(f"正在从 HuggingFace 下载数据集: {dataset_name}...")
    dataset = load_dataset('THUDM/LongBench', dataset_name, split=split)

    data = []
    for item in dataset:
        data.append({
            "input": item.get("input", ""),
            "context": item.get("context", ""),
            "answers": item.get("answers", []),
            "length": item.get("length", 0),
            "dataset": item.get("dataset", dataset_name),
            "language": item.get("language", ""),
            "_id": item.get("_id", "")
        })

    print(f"✓ 加载了 {len(data)} 条数据")
    return data


def load_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    从本地 JSONL 或 JSON 文件加载 LongBench 数据

    Args:
        file_path: 文件路径

    Returns:
        LongBench 格式的数据列表
    """
    print(f"正在从本地文件加载: {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.suffix == '.jsonl':
            data = [json.loads(line) for line in f if line.strip()]
        elif file_path.suffix == '.json':
            content = json.load(f)
            if isinstance(content, list):
                data = content
            elif isinstance(content, dict) and 'data' in content:
                data = content['data']
            else:
                raise ValueError("不支持的 JSON 格式")
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    print(f"✓ 加载了 {len(data)} 条数据")
    return data


def load_from_directory(dir_path: Path) -> List[Dict[str, Any]]:
    """
    从目录中加载所有 JSONL 文件并聚合数据

    Args:
        dir_path: 目录路径

    Returns:
        聚合后的 LongBench 格式数据列表
    """
    print(f"正在扫描目录: {dir_path}")

    # 查找所有 .jsonl 文件
    jsonl_files = list(dir_path.glob("**/*.jsonl"))

    if not jsonl_files:
        print(f"⚠ 警告: 目录 {dir_path} 中未找到任何 .jsonl 文件")
        return []

    print(f"找到 {len(jsonl_files)} 个 JSONL 文件:")
    for f in jsonl_files:
        print(f"  - {f.relative_to(dir_path) if f.is_relative_to(dir_path) else f}")

    # 聚合所有数据
    all_data = []
    for file_path in jsonl_files:
        try:
            data = load_from_file(file_path)
            all_data.extend(data)
        except Exception as e:
            print(f"  ⚠ 跳过文件 {file_path.name}: {e}")

    print(f"\n✓ 从 {len(jsonl_files)} 个文件中总共加载了 {len(all_data)} 条数据")
    return all_data


def deduplicate_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去除重复的数据条目

    Args:
        data: 数据列表

    Returns:
        去重后的数据列表
    """
    seen = set()
    unique_data = []

    for entry in data:
        # 使用 _id 或者整个 entry 的 JSON 字符串作为唯一标识
        if "_id" in entry and entry["_id"]:
            identifier = entry["_id"]
        else:
            # 如果没有 _id，使用 context + input 组合作为唯一标识
            identifier = f"{entry.get('context', '')[:100]}_{entry.get('input', '')[:100]}"

        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(entry)

    duplicates_removed = len(data) - len(unique_data)
    if duplicates_removed > 0:
        print(f"  ℹ 去除了 {duplicates_removed} 条重复数据")

    return unique_data


def convert_longbench_to_benchmarker(
    data: List[Dict[str, Any]],
    num_samples: Optional[int] = None,
    format_type: str = "context_question",
    shuffle: bool = True,
    ensure_unique: bool = True
) -> List[Dict[str, str]]:
    """
    转换 LongBench 数据为 dual_round_benchmarker 格式

    Args:
        data: LongBench 格式的数据列表
        num_samples: 抽取的样本数量 (None 表示全部)
        format_type: 转换格式类型
        shuffle: 是否随机打乱
        ensure_unique: 是否确保数据唯一性（去重）

    Returns:
        dual_round_benchmarker 格式的数据列表
    """
    # 去重
    if ensure_unique:
        data = deduplicate_data(data)

    # 如果请求的样本数超过可用数据，调整为实际可用数量
    if num_samples is not None and num_samples > len(data):
        print(f"  ⚠ 请求的样本数 ({num_samples}) 超过可用数据量 ({len(data)})")
        print(f"  → 将使用所有 {len(data)} 条数据")
        num_samples = len(data)

    if shuffle:
        data = data.copy()
        random.shuffle(data)

    if num_samples is not None and num_samples < len(data):
        data = data[:num_samples]

    converted_data = []
    for entry in data:
        converted_entry = format_longbench_entry(entry, format_type)
        converted_data.append(converted_entry)

    return converted_data


def get_recommended_format(dataset_name: str) -> str:
    """
    根据数据集名称推荐转换格式

    Args:
        dataset_name: LongBench 数据集名称

    Returns:
        推荐的格式类型
    """
    # QA 任务: 需要 context + question
    qa_tasks = [
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
        "hotpotqa", "2wikimqa", "musique", "dureader"
    ]

    # 摘要任务: 只需要 context
    summarization_tasks = [
        "gov_report", "qmsum", "multi_news", "vcsum"
    ]

    # Few-shot 任务: context 包含示例, input 是查询
    fewshot_tasks = [
        "trec", "triviaqa", "samsum", "lsht"
    ]

    # 合成任务
    synthetic_tasks = [
        "passage_count", "passage_retrieval_en", "passage_retrieval_zh"
    ]

    # 代码任务
    code_tasks = [
        "lcc", "repobench-p"
    ]

    if dataset_name in qa_tasks:
        return "context_question"
    elif dataset_name in summarization_tasks:
        return "context_only"
    elif dataset_name in (fewshot_tasks + synthetic_tasks + code_tasks):
        return "concatenate"
    else:
        return "context_question"  # 默认


def main():
    parser = argparse.ArgumentParser(
        description="将 LongBench 数据集转换为 dual_round_benchmarker 格式"
    )

    # 数据源选项
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset",
        type=str,
        nargs='+',
        help="LongBench 数据集名称 (支持多个,从 HuggingFace 下载)"
    )
    source_group.add_argument(
        "--input-file",
        type=Path,
        help="本地 LongBench 数据文件路径 (JSONL 或 JSON 格式)"
    )
    source_group.add_argument(
        "--input-dir",
        type=Path,
        help="本地数据目录路径 (批量转换目录下所有 JSONL 文件)"
    )

    # 转换选项
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="抽取的样本数量 (默认: 全部)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["context_question", "context_only", "question_only", "concatenate", "auto"],
        default="auto",
        help="转换格式 (默认: auto - 根据数据集自动选择)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出文件路径 (JSONL 格式，每行一个 JSON 对象)"
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不随机打乱数据"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 加载数据
    all_data = []

    if args.dataset:
        # 从 HuggingFace 加载
        for dataset_name in args.dataset:
            data = load_from_huggingface(dataset_name)

            # 确定格式类型
            if args.format == "auto":
                format_type = get_recommended_format(dataset_name)
                print(f"使用推荐格式: {format_type}")
            else:
                format_type = args.format

            # 转换数据
            converted = convert_longbench_to_benchmarker(
                data,
                num_samples=args.num_samples if len(args.dataset) == 1 else None,
                format_type=format_type,
                shuffle=not args.no_shuffle
            )
            all_data.extend(converted)

        # 如果加载了多个数据集,在这里进行统一采样
        if len(args.dataset) > 1:
            if args.num_samples and args.num_samples < len(all_data):
                if not args.no_shuffle:
                    random.shuffle(all_data)
                all_data = all_data[:args.num_samples]

    elif args.input_dir:
        # 从目录批量加载所有 JSONL 文件
        data = load_from_directory(args.input_dir)

        # 确定格式类型
        if args.format == "auto":
            # 尝试从第一条数据推断数据集类型
            if data:
                dataset_name = data[0].get("dataset", "")
                format_type = get_recommended_format(dataset_name)
                print(f"检测到数据集类型: {dataset_name}, 使用推荐格式: {format_type}")
            else:
                format_type = "context_question"
                print(f"使用默认格式: {format_type}")
        else:
            format_type = args.format

        # 转换数据
        all_data = convert_longbench_to_benchmarker(
            data,
            num_samples=args.num_samples,
            format_type=format_type,
            shuffle=not args.no_shuffle
        )

    else:
        # 从本地文件加载
        data = load_from_file(args.input_file)

        # 确定格式类型
        if args.format == "auto":
            # 尝试从第一条数据推断
            if data:
                dataset_name = data[0].get("dataset", "")
                format_type = get_recommended_format(dataset_name)
                print(f"检测到数据集: {dataset_name}, 使用推荐格式: {format_type}")
            else:
                format_type = "context_question"
                print(f"使用默认格式: {format_type}")
        else:
            format_type = args.format

        # 转换数据
        all_data = convert_longbench_to_benchmarker(
            data,
            num_samples=args.num_samples,
            format_type=format_type,
            shuffle=not args.no_shuffle
        )

    # 保存结果为 JSONL 格式（每行一个 JSON 对象）
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in all_data:
            # 每行写入一个 JSON 对象，不带缩进
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"转换完成!")
    print(f"{'='*60}")
    print(f"总样本数: {len(all_data)}")
    print(f"输出文件: {args.output}")
    print(f"输出格式: JSONL (每行一个 JSON 对象)")
    print(f"\n使用示例:")
    print(f"python dual_round_benchmarker.py \\")
    print(f"  --dataset {args.output} \\")
    print(f"  --endpoint http://localhost:8000/v1/chat/completions \\")
    print(f"  --num-samples {min(10, len(all_data))} \\")
    print(f"  --concurrency 5")


if __name__ == "__main__":
    main()
