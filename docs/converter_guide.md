# LongBench 数据转换器完整指南

## 目录
- [概述](#概述)
- [安装](#安装)
- [数据格式](#数据格式)
- [使用方法](#使用方法)
- [命令行参数](#命令行参数)
- [支持的数据集](#支持的数据集)
- [转换格式类型](#转换格式类型)
- [核心功能](#核心功能)
- [使用示例](#使用示例)
- [故障排除](#故障排除)

---

## 概述

`convert_longbench.py` 将 LongBench 格式的数据集转换为 `dual_round_benchmarker.py` 可用的 JSONL 格式。

### 核心特性

- ✅ **三种数据源**:
  - HuggingFace 自动下载
  - 本地单个文件
  - 本地目录批量转换
- ✅ **智能去重**: 自动识别并去除重复数据
- ✅ **采样保护**: 防止重复采样
- ✅ **格式推荐**: 根据数据集类型自动选择最佳格式
- ✅ **JSONL 输出**: 每行一个 JSON 对象

---

## 安装

### 基础依赖（必需）

```bash
pip install -r requirements.txt
```

### HuggingFace 支持（可选）

仅在从 HuggingFace 下载数据时需要：

```bash
pip install datasets
```

---

## 数据格式

### 输入格式: LongBench

```json
{
  "input": "What is the name of the main character?",
  "context": "Once upon a time, there was a brave knight named Arthur...",
  "answers": ["Arthur"],
  "length": 1234,
  "dataset": "narrativeqa",
  "_id": "unique_id_123"
}
```

### 输出格式: JSONL

每行一个独立的 JSON 对象：

```
{"text": "Once upon a time, there was a brave knight named Arthur...\n\n问题: What is the name of the main character?"}
{"text": "Another long document here...\n\n问题: Another question?"}
```

**JSONL 特点**:
- ✅ 每行一个完整的 JSON 对象
- ✅ 行之间没有逗号
- ✅ 没有方括号包裹
- ✅ 可逐行流式处理

---

## 使用方法

### 1. 从 HuggingFace 下载并转换

**单个数据集**:
```bash
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 100 \
  --output data/narrativeqa.jsonl
```

**多个数据集混合**:
```bash
python convert_longbench.py \
  --dataset narrativeqa qasper hotpotqa \
  --num-samples 200 \
  --output data/mixed_qa.jsonl
```

### 2. 从本地文件转换

```bash
python convert_longbench.py \
  --input-file LongBench_data/narrativeqa.jsonl \
  --num-samples 100 \
  --output data/narrativeqa.jsonl
```

### 3. 从目录批量转换（推荐）

```bash
python convert_longbench.py \
  --input-dir LongBench_data/ \
  --num-samples 300 \
  --output data/batch_all.jsonl
```

**功能特性**:
- 递归扫描所有 `.jsonl` 文件（包括子目录）
- 自动聚合所有数据
- 自动去重
- 采样保护（不重复采样）

---

## 命令行参数

### 数据源参数（三选一）

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `--dataset` | str[] | HuggingFace 数据集名称 | `--dataset narrativeqa qasper` |
| `--input-file` | Path | 本地文件路径 | `--input-file data/test.jsonl` |
| `--input-dir` | Path | 本地目录路径 | `--input-dir LongBench_data/` |

### 转换参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-samples` | 全部 | 抽取样本数量 |
| `--format` | auto | 转换格式类型 |
| `--output` | **必需** | 输出文件路径 (.jsonl) |
| `--no-shuffle` | False | 不随机打乱数据 |
| `--seed` | 42 | 随机种子 |

---

## 支持的数据集

### 英文数据集

**Single-Document QA**:
- `narrativeqa` - 长篇小说问答
- `qasper` - 学术论文问答
- `multifieldqa_en` - 多领域问答

**Multi-Document QA**:
- `hotpotqa` - 多跳推理
- `2wikimqa` - 维基百科问答
- `musique` - 多步推理

**Summarization**:
- `gov_report` - 政府报告摘要
- `qmsum` - 会议摘要
- `multi_news` - 新闻摘要

**Few-shot Learning**:
- `trec` - 问题分类
- `triviaqa` - 百科知识
- `samsum` - 对话摘要

**Synthetic Tasks**:
- `passage_count` - 段落计数
- `passage_retrieval_en` - 段落检索

**Code**:
- `lcc` - 长代码补全
- `repobench-p` - 代码库补全

### 中文数据集

- `multifieldqa_zh` - 多领域问答
- `dureader` - 百度阅读理解
- `vcsum` - 视频字幕摘要
- `lsht` - 文本分类
- `passage_retrieval_zh` - 段落检索

---

## 转换格式类型

### 1. `context_question` (默认)

适用于 **QA 任务**。

**输入**:
```json
{
  "context": "Arthur was a brave knight...",
  "input": "Who was Arthur?"
}
```

**输出**:
```json
{"text": "Arthur was a brave knight...\n\n问题: Who was Arthur?"}
```

### 2. `context_only`

适用于 **摘要任务**。

**输入**:
```json
{
  "context": "Long government report...",
  "input": "Summarize this report"
}
```

**输出**:
```json
{"text": "Long government report..."}
```

### 3. `concatenate`

适用于 **Few-shot 和合成任务**。

**输入**:
```json
{
  "context": "Example 1: ...\nExample 2: ...",
  "input": "Query text"
}
```

**输出**:
```json
{"text": "Example 1: ...\nExample 2: ...\nQuery text"}
```

### 4. `auto` (推荐)

根据数据集类型自动选择：

| 数据集类型 | 自动选择格式 |
|------------|--------------|
| QA 任务 | `context_question` |
| 摘要任务 | `context_only` |
| Few-shot | `concatenate` |
| 合成任务 | `concatenate` |
| 代码任务 | `concatenate` |

---

## 核心功能

### 1. 智能去重

**去重逻辑**:

```python
# 优先使用 _id
if "_id" in entry and entry["_id"]:
    identifier = entry["_id"]
else:
    # 使用 context + input 前 100 字符
    identifier = f"{context[:100]}_{input[:100]}"
```

**输出示例**:
```
✓ 从 4 个文件中总共加载了 370 条数据
  ℹ 去除了 15 条重复数据
```

### 2. 采样保护

**问题**: 用户请求 500 条样本，但去重后只有 300 条。

**解决方案**: 自动调整为 300 条，**绝不重复采样**。

```
⚠ 请求的样本数 (500) 超过可用数据量 (300)
→ 将使用所有 300 条数据
```

### 3. 目录批量转换

**场景**: 多个 JSONL 文件在同一目录

```
LongBench_data/
├── narrativeqa.jsonl      (100 条)
├── qasper.jsonl           (80 条)
├── hotpotqa.jsonl         (120 条)
└── subfolder/
    └── dureader.jsonl     (70 条)
```

**命令**:
```bash
python convert_longbench.py \
  --input-dir LongBench_data/ \
  --num-samples 200 \
  --output data/mixed.jsonl
```

**输出**:
```
正在扫描目录: LongBench_data/
找到 4 个 JSONL 文件:
  - narrativeqa.jsonl
  - qasper.jsonl
  - hotpotqa.jsonl
  - subfolder/dureader.jsonl

✓ 从 4 个文件中总共加载了 370 条数据
  ℹ 去除了 5 条重复数据

转换完成!
总样本数: 200
输出文件: data/mixed.jsonl
输出格式: JSONL (每行一个 JSON 对象)
```

---

## 使用示例

### 示例 1: 快速测试数据

```bash
# 下载 narrativeqa，取 50 条
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 50 \
  --output data/test.jsonl

# 运行压测
python dual_round_benchmarker.py \
  --dataset data/test.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 20 \
  --concurrency 5
```

### 示例 2: 混合多任务数据集

```bash
# 混合 QA、摘要、推理任务
python convert_longbench.py \
  --dataset narrativeqa gov_report hotpotqa \
  --num-samples 150 \
  --output data/mixed_tasks.jsonl

# 多并发压测
python dual_round_benchmarker.py \
  --dataset data/mixed_tasks.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 100 \
  --concurrency 5 10 20 \
  --output-dir results/mixed
```

### 示例 3: 本地文件批量转换

```bash
# 假设你已经下载了多个 LongBench 数据集到 LongBench_data/
python convert_longbench.py \
  --input-dir LongBench_data/ \
  --num-samples 300 \
  --output data/all_longbench.jsonl

# 长文本压测
python dual_round_benchmarker.py \
  --dataset data/all_longbench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 200 \
  --max-input-length 16384 \
  --concurrency 10 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate \
  --output-dir results/longbench
```

### 示例 4: 中文数据集

```bash
# 中文 QA + 摘要
python convert_longbench.py \
  --dataset dureader vcsum multifieldqa_zh \
  --num-samples 100 \
  --output data/chinese.jsonl

# 测试中文模型
python dual_round_benchmarker.py \
  --dataset data/chinese.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model qwen-7b \
  --num-samples 50 \
  --concurrency 10
```

### 示例 5: 自定义格式

```bash
# 强制使用 context_only 格式
python convert_longbench.py \
  --dataset narrativeqa \
  --format context_only \
  --num-samples 100 \
  --output data/context_only.jsonl

# 不打乱顺序，固定随机种子
python convert_longbench.py \
  --dataset qasper \
  --no-shuffle \
  --seed 123 \
  --num-samples 50 \
  --output data/ordered.jsonl
```

---

## 故障排除

### 问题 1: ModuleNotFoundError: No module named 'datasets'

**错误**:
```
ImportError: 需要安装 datasets 库来从 HuggingFace 下载数据。
运行: pip install datasets
```

**解决方案**:
```bash
# 方案 1: 安装 datasets 库
pip install datasets

# 方案 2: 使用本地文件
python convert_longbench.py \
  --input-file LongBench_data/narrativeqa.jsonl \
  --output data/output.jsonl
```

### 问题 2: HuggingFace 连接超时

**错误**: `Connection timeout` 或下载速度慢

**解决方案**:

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载数据文件后使用 --input-file
wget https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip data.zip -d LongBench_data/
python convert_longbench.py --input-dir LongBench_data/ ...
```

### 问题 3: 目录中未找到 JSONL 文件

**错误**:
```
⚠ 警告: 目录 LongBench_data/ 中未找到任何 .jsonl 文件
```

**解决方案**:

检查目录结构：
```bash
ls -R LongBench_data/
```

确保文件扩展名为 `.jsonl`，不是 `.json` 或其他格式。

### 问题 4: 转换后的文本过长

**问题**: 超过模型的最大输入长度

**解决方案**:

在运行 benchmarker 时截断：
```bash
python dual_round_benchmarker.py \
  --dataset data/output.jsonl \
  --max-input-length 8192 \
  ...
```

### 问题 5: 去重后数据量不足

**场景**:
```
✓ 从 5 个文件中总共加载了 200 条数据
  ℹ 去除了 150 条重复数据
  ⚠ 请求的样本数 (100) 超过可用数据量 (50)
  → 将使用所有 50 条数据
```

**解决方案**:

1. 增加源文件数量
2. 调整 `--num-samples` 为合理值
3. 检查是否有文件重复

---

## 高级用法

### 批量转换所有 LongBench 数据集

```bash
#!/bin/bash
mkdir -p data/longbench

# 所有英文 QA 数据集
qa_datasets=(
    "narrativeqa"
    "qasper"
    "multifieldqa_en"
    "hotpotqa"
    "2wikimqa"
    "musique"
)

# 批量转换
for dataset in "${qa_datasets[@]}"; do
    echo "正在转换: $dataset"
    python convert_longbench.py \
        --dataset "$dataset" \
        --num-samples 100 \
        --output "data/longbench/${dataset}.jsonl"
done

echo "批量转换完成!"
```

### 验证输出格式

```bash
# 验证 JSONL 格式
python tests/test_jsonl_output.py data/output.jsonl

# 输出示例:
# 测试文件: data/output.jsonl
# ✅ JSONL 格式正确!
#    总行数: 100
#    有效对象数: 100
#
# 第一个对象示例:
#   键: ['text']
#   text 预览: Once upon a time...
```

### 手动转换格式（如果需要）

**JSON 数组 → JSONL**:

```python
#!/usr/bin/env python3
import json

# 读取 JSON 数组
with open('old_format.json', 'r') as f:
    data = json.load(f)

# 写入 JSONL
with open('new_format.jsonl', 'w') as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
```

或使用命令行工具 `jq`:

```bash
cat old_format.json | jq -c '.[]' > new_format.jsonl
```

---

## 数据集统计

### 平均长度（tokens）

| 数据集 | 平均长度 | 任务类型 |
|--------|----------|----------|
| narrativeqa | 18K | 长篇小说 QA |
| qasper | 5K | 学术论文 QA |
| gov_report | 9K | 政府报告摘要 |
| hotpotqa | 3K | 多文档 QA |
| multifieldqa | 4K | 多领域 QA |

### 推荐配置

根据数据集平均长度选择参数：

**短文本 (< 2K tokens)**:
```bash
--max-input-length 4096
--timeout 60
```

**中等长度 (2K - 8K tokens)**:
```bash
--max-input-length 8192
--timeout 300
```

**长文本 (> 8K tokens)**:
```bash
--max-input-length 16384
--timeout 600
```

---

## 完整工作流

### 端到端示例

```bash
# 1. 转换数据
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 100 \
  --output data/narrativeqa.jsonl

# 2. 验证格式
python tests/test_jsonl_output.py data/narrativeqa.jsonl

# 3. 小规模测试
python dual_round_benchmarker.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 5 \
  --concurrency 1

# 4. 正式压测
python dual_round_benchmarker.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 20 \
  --max-input-length 8192 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate \
  --slo-file examples/slo_example.yaml \
  --output-dir results/narrativeqa

# 5. 分析结果
cat results/narrativeqa/metrics_summary.csv
```

---

## 参考资料

- [LongBench 论文](https://arxiv.org/abs/2308.14508)
- [LongBench GitHub](https://github.com/THUDM/LongBench)
- [LongBench HuggingFace](https://huggingface.co/datasets/THUDM/LongBench)
- [JSONL 格式规范](http://jsonlines.org/)
