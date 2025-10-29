#!/usr/bin/env python3
"""
Text Dataset Generator - 从纯文本生成FluxPerf可用的数据集

支持三种模式：
1. 单轮固定长度模式 - 生成指定长度的单轮数据集
2. 多轮对话模式 - 生成多轮对话数据集，每轮长度可配置
3. 共享前缀模式 - 生成带有公共前缀的数据集

支持按字符数或按 token 数生成（使用 --use-tokens 选项）
支持顺序无重叠切分（使用 --no-overlap 选项）
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def load_text_corpus(file_path: Path) -> str:
    """加载纯文本语料库"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"[OK] 成功加载语料库: {file_path}")
        print(f"  总字符数: {len(text):,}")
        return text
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            print(f"[OK] 成功加载语料库 (latin-1编码): {file_path}")
            print(f"  总字符数: {len(text):,}")
            return text
        except Exception as e:
            print(f"[ERROR] 加载文件失败: {e}")
            sys.exit(1)


def init_tokenizer(tokenizer_name: str, trust_remote_code: bool = False):
    """
    初始化 tokenizer

    Args:
        tokenizer_name: 模型名称或本地路径
        trust_remote_code: 是否信任远程代码

    Returns:
        tokenizer 对象，如果失败返回 None
    """
    try:
        import contextlib
        import io
        from transformers import AutoTokenizer

        # 静默加载
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=trust_remote_code
            )
        print(f"[OK] Tokenizer 初始化成功: {tokenizer_name}")
        return tokenizer
    except ImportError:
        print(f"[ERROR] 请安装 transformers 库: pip install transformers")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Tokenizer 初始化失败: {e}")
        sys.exit(1)


def count_tokens(text: str, tokenizer) -> int:
    """统计文本的 token 数"""
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_text_by_tokens(text: str, start_token: int, num_tokens: int, tokenizer) -> str:
    """
    从文本中提取指定 token 范围的片段

    Args:
        text: 原始文本
        start_token: 起始 token 位置
        num_tokens: 需要的 token 数量
        tokenizer: tokenizer 对象

    Returns:
        提取的文本片段
    """
    # 编码整个文本
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)

    if start_token + num_tokens > total_tokens:
        num_tokens = total_tokens - start_token

    # 提取指定范围的 token
    selected_tokens = token_ids[start_token:start_token + num_tokens]

    # 解码回文本
    return tokenizer.decode(selected_tokens, skip_special_tokens=True)


def sample_text_segments(
    corpus: str,
    num_samples: int,
    segment_length: int,
    avoid_duplicates: bool = True,
    no_overlap: bool = False,
    use_tokens: bool = False,
    tokenizer = None
) -> List[str]:
    """
    从语料库中随机采样文本片段

    Args:
        corpus: 文本语料库
        num_samples: 采样数量
        segment_length: 每个片段的长度（字符数或 token 数）
        avoid_duplicates: 是否避免重复采样（no_overlap=True 时忽略）
        no_overlap: 是否禁用重叠，按顺序切分
        use_tokens: 是否按 token 数采样（默认按字符数）
        tokenizer: tokenizer 对象（use_tokens=True 时必需）

    Returns:
        采样的文本片段列表
    """
    if use_tokens:
        if tokenizer is None:
            print(f"[ERROR] 使用 token 模式时必须提供 tokenizer")
            sys.exit(1)
        if no_overlap:
            segments, _ = _sample_text_segments_sequential_tokens(
                corpus,
                num_samples,
                segment_length,
                tokenizer,
                start_token=0
            )
            return segments
        return _sample_text_segments_by_tokens(
            corpus, num_samples, segment_length, avoid_duplicates, tokenizer
        )
    else:
        if no_overlap:
            segments, _ = _sample_text_segments_sequential_chars(
                corpus,
                num_samples,
                segment_length,
                start_offset=0
            )
            return segments
        return _sample_text_segments_by_chars(
            corpus, num_samples, segment_length, avoid_duplicates
        )


def _sample_text_segments_by_chars(
    corpus: str,
    num_samples: int,
    segment_length: int,
    avoid_duplicates: bool
) -> List[str]:
    """按字符数采样文本片段"""
    corpus_length = len(corpus)

    if segment_length >= corpus_length:
        print(f"[ERROR] 错误: 片段长度 ({segment_length}) 超过语料库长度 ({corpus_length})")
        sys.exit(1)

    max_start = corpus_length - segment_length

    if avoid_duplicates and num_samples > max_start:
        print(f"[WARNING] 警告: 请求的样本数 ({num_samples}) 超过可能的唯一片段数 ({max_start})")
        print(f"  将生成尽可能多的唯一片段")
        num_samples = min(num_samples, max_start)

    segments = []
    used_starts = set()

    attempts = 0
    max_attempts = num_samples * 10  # 防止无限循环

    while len(segments) < num_samples and attempts < max_attempts:
        start = random.randint(0, max_start)

        if avoid_duplicates and start in used_starts:
            attempts += 1
            continue

        segment = corpus[start:start + segment_length]
        segments.append(segment)

        if avoid_duplicates:
            used_starts.add(start)

        attempts += 1

    if len(segments) < num_samples:
        print(f"[WARNING] 警告: 仅生成了 {len(segments)} 个唯一片段（请求 {num_samples} 个）")

    return segments


def _sample_text_segments_sequential_chars(
    corpus: str,
    num_samples: int,
    segment_length: int,
    start_offset: int = 0
) -> Tuple[List[str], int]:
    """按字符顺序切分非重叠片段"""
    corpus_length = len(corpus)

    if segment_length <= 0:
        print(f"[ERROR] 错误: 片段长度必须大于 0")
        sys.exit(1)

    if start_offset >= corpus_length:
        print(f"[WARNING] 警告: 起始位置 ({start_offset}) 超出语料库长度 ({corpus_length})")
        return [], corpus_length

    remaining = corpus_length - start_offset
    max_samples = remaining // segment_length

    if max_samples == 0:
        print(f"[WARNING] 警告: 剩余字符不足以生成一个长度为 {segment_length} 的片段")
        return [], corpus_length

    if num_samples > max_samples:
        print(f"[WARNING] 警告: 请求的样本数 ({num_samples}) 超过可用的不重叠片段数 ({max_samples})")
        num_samples = max_samples

    segments = []
    current = start_offset

    for _ in range(num_samples):
        end = current + segment_length
        segments.append(corpus[current:end])
        current = end

    return segments, current


def _sample_text_segments_by_tokens(
    corpus: str,
    num_samples: int,
    num_tokens: int,
    avoid_duplicates: bool,
    tokenizer
) -> List[str]:
    """按 token 数采样文本片段"""
    # 编码整个语料库
    print(f"[INFO] 正在 tokenize 语料库...")
    corpus_tokens = tokenizer.encode(corpus, add_special_tokens=False)
    total_tokens = len(corpus_tokens)
    print(f"[OK] 语料库共有 {total_tokens:,} tokens")

    if num_tokens >= total_tokens:
        print(f"[ERROR] 错误: 片段长度 ({num_tokens} tokens) 超过语料库长度 ({total_tokens} tokens)")
        sys.exit(1)

    max_start = total_tokens - num_tokens

    if avoid_duplicates and num_samples > max_start:
        print(f"[WARNING] 警告: 请求的样本数 ({num_samples}) 超过可能的唯一片段数 ({max_start})")
        print(f"  将生成尽可能多的唯一片段")
        num_samples = min(num_samples, max_start)

    segments = []
    used_starts = set()

    attempts = 0
    max_attempts = num_samples * 10  # 防止无限循环

    print(f"[INFO] 正在采样 {num_samples} 个片段...")
    while len(segments) < num_samples and attempts < max_attempts:
        start_token = random.randint(0, max_start)

        if avoid_duplicates and start_token in used_starts:
            attempts += 1
            continue

        # 提取 token 范围并解码
        selected_tokens = corpus_tokens[start_token:start_token + num_tokens]
        segment = tokenizer.decode(selected_tokens, skip_special_tokens=True)
        segments.append(segment)

        if avoid_duplicates:
            used_starts.add(start_token)

        attempts += 1

        # 进度显示
        if len(segments) % max(1, num_samples // 10) == 0:
            print(f"  进度: {len(segments)}/{num_samples}")

    if len(segments) < num_samples:
        print(f"[WARNING] 警告: 仅生成了 {len(segments)} 个唯一片段（请求 {num_samples} 个）")

    return segments


def _sample_text_segments_sequential_tokens(
    corpus: str,
    num_samples: int,
    num_tokens: int,
    tokenizer,
    start_token: int = 0
) -> Tuple[List[str], int]:
    """按 token 顺序切分非重叠片段"""
    if num_tokens <= 0:
        print(f"[ERROR] 错误: 片段 token 数必须大于 0")
        sys.exit(1)

    print(f"[INFO] 正在 tokenize 语料库...")
    corpus_tokens = tokenizer.encode(corpus, add_special_tokens=False)
    total_tokens = len(corpus_tokens)
    print(f"[OK] 语料库共有 {total_tokens:,} tokens")

    if start_token >= total_tokens:
        print(f"[WARNING] 警告: 起始 token ({start_token}) 超出语料库总 token 数 ({total_tokens})")
        return [], total_tokens

    remaining = total_tokens - start_token
    max_samples = remaining // num_tokens

    if max_samples == 0:
        print(f"[WARNING] 警告: 剩余 token 不足以生成一个长度为 {num_tokens} 的片段")
        return [], total_tokens

    if num_samples > max_samples:
        print(f"[WARNING] 警告: 请求的样本数 ({num_samples}) 超过可用的不重叠片段数 ({max_samples})")
        num_samples = max_samples

    segments = []
    current = start_token

    for _ in range(num_samples):
        end = current + num_tokens
        selected_tokens = corpus_tokens[current:end]
        segment = tokenizer.decode(selected_tokens, skip_special_tokens=True)
        segments.append(segment)
        current = end

    return segments, current


def generate_single_turn_dataset(
    corpus: str,
    num_entries: int,
    entry_length: int,
    output_file: Path,
    no_overlap: bool = False,
    use_tokens: bool = False,
    tokenizer = None
):
    """
    模式1: 生成单轮固定长度数据集

    Args:
        corpus: 文本语料库
        num_entries: 数据集条目数
        entry_length: 每个条目的长度（字符数或 token 数）
        output_file: 输出文件路径
        use_tokens: 是否按 token 数生成
        tokenizer: tokenizer 对象
    """
    print(f"\n{'='*60}")
    print(f"模式1: 生成单轮固定长度数据集")
    print(f"{'='*60}")
    print(f"条目数: {num_entries}")
    unit = "tokens" if use_tokens else "字符"
    print(f"每条长度: {entry_length} {unit}")

    segments = sample_text_segments(
        corpus, num_entries, entry_length,
        no_overlap=no_overlap,
        use_tokens=use_tokens, tokenizer=tokenizer
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            entry = {
                "id": f"entry_{i}",
                "text": segment
            }
            if use_tokens and tokenizer:
                entry["token_count"] = count_tokens(segment, tokenizer)
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n[OK] 数据集已生成: {output_file}")
    print(f"  总条目数: {len(segments)}")
    print(f"  文件大小: {output_file.stat().st_size / 1024:.2f} KB")


def generate_multi_turn_dataset(
    corpus: str,
    num_conversations: int,
    num_turns: int,
    turn_length: int,
    output_file: Path,
    no_overlap: bool = False,
    use_tokens: bool = False,
    tokenizer = None
):
    """
    模式2: 生成多轮对话数据集

    Args:
        corpus: 文本语料库
        num_conversations: 对话数量
        num_turns: 每个对话的轮次
        turn_length: 每轮的长度（字符数或 token 数）
        output_file: 输出文件路径
        use_tokens: 是否按 token 数生成
        tokenizer: tokenizer 对象
    """
    print(f"\n{'='*60}")
    print(f"模式2: 生成多轮对话数据集")
    print(f"{'='*60}")
    print(f"对话数: {num_conversations}")
    print(f"每对话轮次: {num_turns}")
    unit = "tokens" if use_tokens else "字符"
    print(f"每轮长度: {turn_length} {unit}")

    total_segments_needed = num_conversations * num_turns
    segments = sample_text_segments(
        corpus, total_segments_needed, turn_length,
        no_overlap=no_overlap,
        use_tokens=use_tokens, tokenizer=tokenizer
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        for conv_idx in range(num_conversations):
            user_messages = []
            for turn_idx in range(num_turns):
                segment_idx = conv_idx * num_turns + turn_idx
                if segment_idx < len(segments):
                    user_messages.append(segments[segment_idx])

            entry = {
                "id": f"conversation_{conv_idx}",
                "user_messages": user_messages
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n[OK] 数据集已生成: {output_file}")
    print(f"  总对话数: {num_conversations}")
    print(f"  总轮次数: {num_conversations * num_turns}")
    print(f"  文件大小: {output_file.stat().st_size / 1024:.2f} KB")


def generate_common_prefix_dataset(
    corpus: str,
    num_entries: int,
    total_length: int,
    prefix_length: int,
    output_file: Path,
    no_overlap: bool = False,
    use_tokens: bool = False,
    tokenizer = None
):
    """
    模式3: 生成带有公共前缀的数据集

    Args:
        corpus: 文本语料库
        num_entries: 数据集条目数
        total_length: 每个条目的总长度（字符数或 token 数）
        prefix_length: 公共前缀的长度（字符数或 token 数）
        output_file: 输出文件路径
        use_tokens: 是否按 token 数生成
        tokenizer: tokenizer 对象
    """
    print(f"\n{'='*60}")
    print(f"模式3: 生成共享前缀数据集")
    print(f"{'='*60}")
    print(f"条目数: {num_entries}")
    unit = "tokens" if use_tokens else "字符"
    print(f"总长度: {total_length} {unit}")
    print(f"公共前缀长度: {prefix_length} {unit}")
    print(f"随机后缀长度: {total_length - prefix_length} {unit}")

    if prefix_length >= total_length:
        print(f"[ERROR] 错误: 前缀长度 ({prefix_length}) 必须小于总长度 ({total_length})")
        sys.exit(1)

    # 生成公共前缀
    if no_overlap:
        if use_tokens:
            prefix_segments, next_offset = _sample_text_segments_sequential_tokens(
                corpus,
                1,
                prefix_length,
                tokenizer,
                start_token=0
            )
        else:
            prefix_segments, next_offset = _sample_text_segments_sequential_chars(
                corpus,
                1,
                prefix_length,
                start_offset=0
            )
        if not prefix_segments:
            print(f"[ERROR] 错误: 无法生成公共前缀，请检查语料库长度")
            sys.exit(1)
        common_prefix = prefix_segments[0]
    else:
        common_prefix = sample_text_segments(
            corpus, 1, prefix_length,
            no_overlap=no_overlap,
            use_tokens=use_tokens, tokenizer=tokenizer
        )[0]
    print(f"\n[OK] 已生成公共前缀:")
    print(f"  前100字符: {common_prefix[:100]}...")

    # 生成随机后缀
    suffix_length = total_length - prefix_length
    if no_overlap:
        if use_tokens:
            suffixes, _ = _sample_text_segments_sequential_tokens(
                corpus,
                num_entries,
                suffix_length,
                tokenizer,
                start_token=next_offset
            )
        else:
            suffixes, _ = _sample_text_segments_sequential_chars(
                corpus,
                num_entries,
                suffix_length,
                start_offset=next_offset
            )
    else:
        suffixes = sample_text_segments(
            corpus, num_entries, suffix_length,
            no_overlap=no_overlap,
            use_tokens=use_tokens, tokenizer=tokenizer
        )

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, suffix in enumerate(suffixes):
            full_text = common_prefix + suffix
            entry = {
                "id": f"prefix_entry_{i}",
                "text": full_text,
                "prefix_length": prefix_length,
                "suffix_length": suffix_length
            }
            if use_tokens and tokenizer:
                entry["prefix_token_count"] = count_tokens(common_prefix, tokenizer)
                entry["suffix_token_count"] = count_tokens(suffix, tokenizer)
                entry["total_token_count"] = count_tokens(full_text, tokenizer)
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n[OK] 数据集已生成: {output_file}")
    print(f"  总条目数: {len(suffixes)}")
    print(f"  文件大小: {output_file.stat().st_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="从纯文本生成FluxPerf可用的JSONL数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 生成单轮固定长度数据集:
   python text_dataset_generator.py \\
     --input shakespeare.txt \\
     --output dataset.jsonl \\
     --mode single \\
     --num-entries 600 \\
     --length 1000

2. 生成多轮对话数据集:
   python text_dataset_generator.py \\
     --input harry_potter.txt \\
     --output multi_turn.jsonl \\
     --mode multi \\
     --num-conversations 100 \\
     --num-turns 5 \\
     --turn-length 500

3. 生成共享前缀数据集:
   python text_dataset_generator.py \\
     --input corpus.txt \\
     --output prefix_dataset.jsonl \\
     --mode prefix \\
     --num-entries 200 \\
     --total-length 2000 \\
     --prefix-length 1000
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='输入的纯文本文件路径'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='输出的JSONL文件路径'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'multi', 'prefix'],
        required=True,
        help='生成模式: single(单轮), multi(多轮), prefix(共享前缀)'
    )

    # 模式1参数
    parser.add_argument(
        '--num-entries',
        type=int,
        help='[模式1/3] 数据集条目数'
    )

    parser.add_argument(
        '--length',
        type=int,
        help='[模式1] 每个条目的长度（字符数）'
    )

    # 模式2参数
    parser.add_argument(
        '--num-conversations',
        type=int,
        help='[模式2] 对话数量'
    )

    parser.add_argument(
        '--num-turns',
        type=int,
        help='[模式2] 每个对话的轮次'
    )

    parser.add_argument(
        '--turn-length',
        type=int,
        help='[模式2] 每轮的长度（字符数）'
    )

    # 模式3参数
    parser.add_argument(
        '--total-length',
        type=int,
        help='[模式3] 每个条目的总长度（字符数）'
    )

    parser.add_argument(
        '--prefix-length',
        type=int,
        help='[模式3] 公共前缀的长度（字符数）'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子，用于可重复性'
    )

    parser.add_argument(
        '--no-overlap',
        action='store_true',
        help='禁用段落重叠，按顺序切分片段'
    )

    # Tokenizer 相关参数
    parser.add_argument(
        '--use-tokens',
        action='store_true',
        help='按 token 数生成数据集（而不是字符数），需要指定 --tokenizer'
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        help='Tokenizer 名称或本地路径（例如：Qwen/Qwen2.5-7B-Instruct）'
    )

    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='信任远程代码（某些模型如 Qwen/ChatGLM 需要此选项）'
    )

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")

    # 初始化 tokenizer（如果使用 token 模式）
    tokenizer = None
    if args.use_tokens:
        if not args.tokenizer:
            parser.error("--use-tokens 需要指定 --tokenizer 参数")
        tokenizer = init_tokenizer(args.tokenizer, args.trust_remote_code)

    # 加载语料库
    corpus = load_text_corpus(args.input)

    # 根据模式生成数据集
    if args.mode == 'single':
        if not args.num_entries or not args.length:
            parser.error("模式1需要 --num-entries 和 --length 参数")
        generate_single_turn_dataset(
            corpus,
            args.num_entries,
            args.length,
            args.output,
            no_overlap=args.no_overlap,
            use_tokens=args.use_tokens,
            tokenizer=tokenizer
        )

    elif args.mode == 'multi':
        if not args.num_conversations or not args.num_turns or not args.turn_length:
            parser.error("模式2需要 --num-conversations, --num-turns, 和 --turn-length 参数")
        generate_multi_turn_dataset(
            corpus,
            args.num_conversations,
            args.num_turns,
            args.turn_length,
            args.output,
            no_overlap=args.no_overlap,
            use_tokens=args.use_tokens,
            tokenizer=tokenizer
        )

    elif args.mode == 'prefix':
        if not args.num_entries or not args.total_length or not args.prefix_length:
            parser.error("模式3需要 --num-entries, --total-length, 和 --prefix-length 参数")
        generate_common_prefix_dataset(
            corpus,
            args.num_entries,
            args.total_length,
            args.prefix_length,
            args.output,
            no_overlap=args.no_overlap,
            use_tokens=args.use_tokens,
            tokenizer=tokenizer
        )

    print(f"\n[OK] 完成！数据集可直接用于FluxPerf测试")


if __name__ == '__main__':
    main()
