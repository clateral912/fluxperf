#!/usr/bin/env python3
"""
LongBench Dataset Conversion Tool

Converts LongBench dataset format to JSONL format usable by fluxperf.
Supports downloading from HuggingFace or loading from local JSONL files.

Output format: JSONL (one JSON object per line)

Usage:
    # Download from HuggingFace and convert
    python convert_longbench.py --dataset narrativeqa --num-samples 100 --output narrativeqa_bench.jsonl

    # Convert from local file
    python convert_longbench.py --input-file data/narrativeqa.jsonl --num-samples 100 --output narrativeqa_bench.jsonl

    # Batch convert all JSONL files from directory
    python convert_longbench.py --input-dir LongBench_data/ --num-samples 200 --output mixed_bench.jsonl

    # Convert multiple datasets
    python convert_longbench.py --dataset narrativeqa qasper hotpotqa --num-samples 50 --output longbench_mixed.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


def format_longbench_entry(entry: Dict[str, Any], format_type: str = "context_question") -> Dict[str, str]:
    """
    Convert a single LongBench format entry to fluxperf format

    Args:
        entry: LongBench format data entry
        format_type: Conversion format type
            - "context_question": Combine context and input into Q&A format
            - "context_only": Use only context (for summarization tasks)
            - "question_only": Use only input (for tasks without long context)
            - "concatenate": Simple concatenation of context and input

    Returns:
        Dictionary in fluxperf format
    """
    context = entry.get("context", "")
    input_text = entry.get("input", "")

    if format_type == "context_question":
        # Q&A format: provide document first, then question
        if context and input_text:
            combined_text = f"{context}\n\nQuestion: {input_text}"
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
        raise ValueError(f"Unknown format type: {format_type}")

    return {"text": combined_text}


def load_from_huggingface(dataset_name: str, split: str = "test") -> List[Dict[str, Any]]:
    """
    Load LongBench dataset from HuggingFace

    Args:
        dataset_name: Dataset name (e.g., "narrativeqa", "qasper", etc.)
        split: Dataset split (default "test")

    Returns:
        List of data in LongBench format
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library is required to download data from HuggingFace.\n"
            "Run: pip install datasets"
        )

    print(f"Downloading dataset from HuggingFace: {dataset_name}...")
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

    print(f"✓ Loaded {len(data)} entries")
    return data


def load_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load LongBench data from local JSONL or JSON file

    Args:
        file_path: File path

    Returns:
        List of data in LongBench format
    """
    print(f"Loading from local file: {file_path}...")

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
                raise ValueError("Unsupported JSON format")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    print(f"✓ Loaded {len(data)} entries")
    return data


def load_from_directory(dir_path: Path) -> List[Dict[str, Any]]:
    """
    Load and aggregate all JSONL files from directory

    Args:
        dir_path: Directory path

    Returns:
        Aggregated list of data in LongBench format
    """
    print(f"Scanning directory: {dir_path}")

    # Find all .jsonl files
    jsonl_files = list(dir_path.glob("**/*.jsonl"))

    if not jsonl_files:
        print(f"⚠ Warning: No .jsonl files found in directory {dir_path}")
        return []

    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f.relative_to(dir_path) if f.is_relative_to(dir_path) else f}")

    # Aggregate all data
    all_data = []
    for file_path in jsonl_files:
        try:
            data = load_from_file(file_path)
            all_data.extend(data)
        except Exception as e:
            print(f"  ⚠ Skipping file {file_path.name}: {e}")

    print(f"\n✓ Loaded a total of {len(all_data)} entries from {len(jsonl_files)} files")
    return all_data


def deduplicate_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate data entries

    Args:
        data: Data list

    Returns:
        Deduplicated data list
    """
    seen = set()
    unique_data = []

    for entry in data:
        # Use _id or JSON string of entire entry as unique identifier
        if "_id" in entry and entry["_id"]:
            identifier = entry["_id"]
        else:
            # If no _id, use context + input combination as unique identifier
            identifier = f"{entry.get('context', '')[:100]}_{entry.get('input', '')[:100]}"

        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(entry)

    duplicates_removed = len(data) - len(unique_data)
    if duplicates_removed > 0:
        print(f"  ℹ Removed {duplicates_removed} duplicate entries")

    return unique_data


def convert_longbench_to_benchmarker(
    data: List[Dict[str, Any]],
    num_samples: Optional[int] = None,
    format_type: str = "context_question",
    shuffle: bool = True,
    ensure_unique: bool = True
) -> List[Dict[str, str]]:
    """
    Convert LongBench data to fluxperf format

    Args:
        data: List of data in LongBench format
        num_samples: Number of samples to extract (None means all)
        format_type: Conversion format type
        shuffle: Whether to randomly shuffle
        ensure_unique: Whether to ensure data uniqueness (deduplication)

    Returns:
        List of data in fluxperf format
    """
    # Deduplication
    if ensure_unique:
        data = deduplicate_data(data)

    # If requested sample count exceeds available data, adjust to actual available count
    if num_samples is not None and num_samples > len(data):
        print(f"  ⚠ Requested sample count ({num_samples}) exceeds available data ({len(data)})")
        print(f"  → Will use all {len(data)} entries")
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
    Recommend conversion format based on dataset name

    Args:
        dataset_name: LongBench dataset name

    Returns:
        Recommended format type
    """
    # QA tasks: require context + question
    qa_tasks = [
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
        "hotpotqa", "2wikimqa", "musique", "dureader"
    ]

    # Summarization tasks: only need context
    summarization_tasks = [
        "gov_report", "qmsum", "multi_news", "vcsum"
    ]

    # Few-shot tasks: context contains examples, input is query
    fewshot_tasks = [
        "trec", "triviaqa", "samsum", "lsht"
    ]

    # Synthetic tasks
    synthetic_tasks = [
        "passage_count", "passage_retrieval_en", "passage_retrieval_zh"
    ]

    # Code tasks
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
        return "context_question"  # default


def main():
    parser = argparse.ArgumentParser(
        description="Convert LongBench dataset to fluxperf format"
    )

    # Data source options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset",
        type=str,
        nargs='+',
        help="LongBench dataset name(s) (supports multiple, downloads from HuggingFace)"
    )
    source_group.add_argument(
        "--input-file",
        type=Path,
        help="Local LongBench data file path (JSONL or JSON format)"
    )
    source_group.add_argument(
        "--input-dir",
        type=Path,
        help="Local data directory path (batch convert all JSONL files in directory)"
    )

    # Conversion options
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to extract (default: all)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["context_question", "context_only", "question_only", "concatenate", "auto"],
        default="auto",
        help="Conversion format (default: auto - automatically select based on dataset)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path (JSONL format, one JSON object per line)"
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not randomly shuffle data"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load data
    all_data = []

    if args.dataset:
        # Load from HuggingFace
        for dataset_name in args.dataset:
            data = load_from_huggingface(dataset_name)

            # Determine format type
            if args.format == "auto":
                format_type = get_recommended_format(dataset_name)
                print(f"Using recommended format: {format_type}")
            else:
                format_type = args.format

            # Convert data
            converted = convert_longbench_to_benchmarker(
                data,
                num_samples=args.num_samples if len(args.dataset) == 1 else None,
                format_type=format_type,
                shuffle=not args.no_shuffle
            )
            all_data.extend(converted)

        # If multiple datasets loaded, perform unified sampling here
        if len(args.dataset) > 1:
            if args.num_samples and args.num_samples < len(all_data):
                if not args.no_shuffle:
                    random.shuffle(all_data)
                all_data = all_data[:args.num_samples]

    elif args.input_dir:
        # Batch load all JSONL files from directory
        data = load_from_directory(args.input_dir)

        # Determine format type
        if args.format == "auto":
            # Try to infer dataset type from first entry
            if data:
                dataset_name = data[0].get("dataset", "")
                format_type = get_recommended_format(dataset_name)
                print(f"Detected dataset type: {dataset_name}, using recommended format: {format_type}")
            else:
                format_type = "context_question"
                print(f"Using default format: {format_type}")
        else:
            format_type = args.format

        # Convert data
        all_data = convert_longbench_to_benchmarker(
            data,
            num_samples=args.num_samples,
            format_type=format_type,
            shuffle=not args.no_shuffle
        )

    else:
        # Load from local file
        data = load_from_file(args.input_file)

        # Determine format type
        if args.format == "auto":
            # Try to infer from first entry
            if data:
                dataset_name = data[0].get("dataset", "")
                format_type = get_recommended_format(dataset_name)
                print(f"Detected dataset: {dataset_name}, using recommended format: {format_type}")
            else:
                format_type = "context_question"
                print(f"Using default format: {format_type}")
        else:
            format_type = args.format

        # Convert data
        all_data = convert_longbench_to_benchmarker(
            data,
            num_samples=args.num_samples,
            format_type=format_type,
            shuffle=not args.no_shuffle
        )

    # Save results as JSONL format (one JSON object per line)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for entry in all_data:
            # Write one JSON object per line, without indentation
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_data)}")
    print(f"Output file: {args.output}")
    print(f"Output format: JSONL (one JSON object per line)")
    print(f"\nUsage example:")
    print(f"python fluxperf.py \\")
    print(f"  --dataset {args.output} \\")
    print(f"  --endpoint http://localhost:8000/v1/chat/completions \\")
    print(f"  --num-samples {min(10, len(all_data))} \\")
    print(f"  --concurrency 5")


if __name__ == "__main__":
    main()
