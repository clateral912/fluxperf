# LongBench Data Converter Complete Guide

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Formats](#data-formats)
- [Usage](#usage)
- [Command-Line Arguments](#command-line-arguments)
- [Supported Datasets](#supported-datasets)
- [Conversion Format Types](#conversion-format-types)
- [Core Features](#core-features)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

`convert_longbench.py` converts LongBench format datasets into JSONL format usable by `fluxperf.py`.

### Core Features

- ✅ **Three Data Sources**:
  - HuggingFace automatic download
  - Local single file
  - Local directory batch conversion
- ✅ **Smart Deduplication**: Automatically identifies and removes duplicate data
- ✅ **Sampling Protection**: Prevents duplicate sampling
- ✅ **Format Recommendation**: Automatically selects best format based on dataset type
- ✅ **JSONL Output**: One JSON object per line

---

## Installation

### Basic Dependencies (Required)

```bash
pip install -r requirements.txt
```

### HuggingFace Support (Optional)

Only required when downloading data from HuggingFace:

```bash
pip install datasets
```

---

## Data Formats

### Input Format: LongBench

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

### Output Format: JSONL

One independent JSON object per line:

```
{"text": "Once upon a time, there was a brave knight named Arthur...\n\nQuestion: What is the name of the main character?"}
{"text": "Another long document here...\n\nQuestion: Another question?"}
```

**JSONL Characteristics**:
- ✅ One complete JSON object per line
- ✅ No commas between lines
- ✅ No wrapping square brackets
- ✅ Can process line by line in streaming fashion

---

## Usage

### 1. Download and Convert from HuggingFace

**Single Dataset**:
```bash
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 100 \
  --output data/narrativeqa.jsonl
```

**Multiple Datasets Mixed**:
```bash
python convert_longbench.py \
  --dataset narrativeqa qasper hotpotqa \
  --num-samples 200 \
  --output data/mixed_qa.jsonl
```

### 2. Convert from Local File

```bash
python convert_longbench.py \
  --input-file LongBench_data/narrativeqa.jsonl \
  --num-samples 100 \
  --output data/narrativeqa.jsonl
```

### 3. Batch Convert from Directory (Recommended)

```bash
python convert_longbench.py \
  --input-dir LongBench_data/ \
  --num-samples 300 \
  --output data/batch_all.jsonl
```

**Features**:
- Recursively scans all `.jsonl` files (including subdirectories)
- Automatically aggregates all data
- Automatic deduplication
- Sampling protection (no duplicate sampling)

---

## Command-Line Arguments

### Data Source Arguments (Choose One)

| Argument | Type | Description | Example |
|------|------|------|------|
| `--dataset` | str[] | HuggingFace dataset name | `--dataset narrativeqa qasper` |
| `--input-file` | Path | Local file path | `--input-file data/test.jsonl` |
| `--input-dir` | Path | Local directory path | `--input-dir LongBench_data/` |

### Conversion Arguments

| Argument | Default | Description |
|------|--------|------|
| `--num-samples` | All | Number of samples to extract |
| `--format` | auto | Conversion format type |
| `--output` | **Required** | Output file path (.jsonl) |
| `--no-shuffle` | False | Don't randomly shuffle data |
| `--seed` | 42 | Random seed |

---

## Supported Datasets

### English Datasets

**Single-Document QA**:
- `narrativeqa` - Long novel Q&A
- `qasper` - Academic paper Q&A
- `multifieldqa_en` - Multi-field Q&A

**Multi-Document QA**:
- `hotpotqa` - Multi-hop reasoning
- `2wikimqa` - Wikipedia Q&A
- `musique` - Multi-step reasoning

**Summarization**:
- `gov_report` - Government report summary
- `qmsum` - Meeting summary
- `multi_news` - News summary

**Few-shot Learning**:
- `trec` - Question classification
- `triviaqa` - Encyclopedic knowledge
- `samsum` - Dialogue summary

**Synthetic Tasks**:
- `passage_count` - Passage counting
- `passage_retrieval_en` - Passage retrieval

**Code**:
- `lcc` - Long code completion
- `repobench-p` - Repository completion

### Chinese Datasets

- `multifieldqa_zh` - Multi-field Q&A
- `dureader` - Baidu Reading Comprehension
- `vcsum` - Video caption summary
- `lsht` - Text classification
- `passage_retrieval_zh` - Passage retrieval

---

## Conversion Format Types

### 1. `context_question` (Default)

Suitable for **QA Tasks**.

**Input**:
```json
{
  "context": "Arthur was a brave knight...",
  "input": "Who was Arthur?"
}
```

**Output**:
```json
{"text": "Arthur was a brave knight...\n\nQuestion: Who was Arthur?"}
```

### 2. `context_only`

Suitable for **Summarization Tasks**.

**Input**:
```json
{
  "context": "Long government report...",
  "input": "Summarize this report"
}
```

**Output**:
```json
{"text": "Long government report..."}
```

### 3. `concatenate`

Suitable for **Few-shot and Synthetic Tasks**.

**Input**:
```json
{
  "context": "Example 1: ...\nExample 2: ...",
  "input": "Query text"
}
```

**Output**:
```json
{"text": "Example 1: ...\nExample 2: ...\nQuery text"}
```

### 4. `auto` (Recommended)

Automatically selects based on dataset type:

| Dataset Type | Auto-Selected Format |
|------------|--------------|
| QA Tasks | `context_question` |
| Summarization Tasks | `context_only` |
| Few-shot | `concatenate` |
| Synthetic Tasks | `concatenate` |
| Code Tasks | `concatenate` |

---

## Core Features

### 1. Smart Deduplication

**Deduplication Logic**:

```python
# Prioritize using _id
if "_id" in entry and entry["_id"]:
    identifier = entry["_id"]
else:
    # Use first 100 characters of context + input
    identifier = f"{context[:100]}_{input[:100]}"
```

**Output Example**:
```
✓ Loaded a total of 370 entries from 4 files
  ℹ Removed 15 duplicate entries
```

### 2. Sampling Protection

**Problem**: User requests 500 samples, but only 300 remain after deduplication.

**Solution**: Automatically adjust to 300, **never duplicate sampling**.

```
⚠ Requested sample count (500) exceeds available data (300)
→ Will use all 300 entries
```

### 3. Directory Batch Conversion

**Scenario**: Multiple JSONL files in same directory

```
LongBench_data/
├── narrativeqa.jsonl      (100 entries)
├── qasper.jsonl           (80 entries)
├── hotpotqa.jsonl         (120 entries)
└── subfolder/
    └── dureader.jsonl     (70 entries)
```

**Command**:
```bash
python convert_longbench.py \
  --input-dir LongBench_data/ \
  --num-samples 200 \
  --output data/mixed.jsonl
```

**Output**:
```
Scanning directory: LongBench_data/
Found 4 JSONL files:
  - narrativeqa.jsonl
  - qasper.jsonl
  - hotpotqa.jsonl
  - subfolder/dureader.jsonl

✓ Loaded a total of 370 entries from 4 files
  ℹ Removed 5 duplicate entries

Conversion complete!
Total samples: 200
Output file: data/mixed.jsonl
Output format: JSONL (one JSON object per line)
```

---

## Usage Examples

### Example 1: Quick Test Data

```bash
# Download narrativeqa, take 50 entries
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 50 \
  --output data/test.jsonl

# Run stress test
python fluxperf.py \
  --dataset data/test.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 20 \
  --concurrency 5
```

### Example 2: Mixed Multi-Task Dataset

```bash
# Mix QA, summary, reasoning tasks
python convert_longbench.py \
  --dataset narrativeqa gov_report hotpotqa \
  --num-samples 150 \
  --output data/mixed_tasks.jsonl

# Multi-concurrency stress test
python fluxperf.py \
  --dataset data/mixed_tasks.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 100 \
  --concurrency 5 10 20 \
  --output-dir results/mixed
```

### Example 3: Batch Convert Local Files

```bash
# Assuming you've downloaded multiple LongBench datasets to LongBench_data/
python convert_longbench.py \
  --input-dir LongBench_data/ \
  --num-samples 300 \
  --output data/all_longbench.jsonl

# Long text stress test
python fluxperf.py \
  --dataset data/all_longbench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 200 \
  --max-input-length 16384 \
  --concurrency 10 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate \
  --output-dir results/longbench
```

### Example 4: Chinese Datasets

```bash
# Chinese QA + summary
python convert_longbench.py \
  --dataset dureader vcsum multifieldqa_zh \
  --num-samples 100 \
  --output data/chinese.jsonl

# Test Chinese model
python fluxperf.py \
  --dataset data/chinese.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model qwen-7b \
  --num-samples 50 \
  --concurrency 10
```

### Example 5: Custom Format

```bash
# Force use of context_only format
python convert_longbench.py \
  --dataset narrativeqa \
  --format context_only \
  --num-samples 100 \
  --output data/context_only.jsonl

# Don't shuffle, fixed random seed
python convert_longbench.py \
  --dataset qasper \
  --no-shuffle \
  --seed 123 \
  --num-samples 50 \
  --output data/ordered.jsonl
```

---

## Troubleshooting

### Issue 1: ModuleNotFoundError: No module named 'datasets'

**Error**:
```
ImportError: Need to install datasets library to download from HuggingFace.
Run: pip install datasets
```

**Solution**:
```bash
# Option 1: Install datasets library
pip install datasets

# Option 2: Use local file
python convert_longbench.py \
  --input-file LongBench_data/narrativeqa.jsonl \
  --output data/output.jsonl
```

### Issue 2: HuggingFace Connection Timeout

**Error**: `Connection timeout` or slow download

**Solution**:

```bash
# Set HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com

# Or manually download data files then use --input-file
wget https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip data.zip -d LongBench_data/
python convert_longbench.py --input-dir LongBench_data/ ...
```

### Issue 3: No JSONL Files Found in Directory

**Error**:
```
⚠ Warning: No .jsonl files found in directory LongBench_data/
```

**Solution**:

Check directory structure:
```bash
ls -R LongBench_data/
```

Ensure file extension is `.jsonl`, not `.json` or other formats.

### Issue 4: Converted Text Too Long

**Problem**: Exceeds model's maximum input length

**Solution**:

Truncate when running benchmarker:
```bash
python fluxperf.py \
  --dataset data/output.jsonl \
  --max-input-length 8192 \
  ...
```

### Issue 5: Insufficient Data After Deduplication

**Scenario**:
```
✓ Loaded a total of 200 entries from 5 files
  ℹ Removed 150 duplicate entries
  ⚠ Requested sample count (100) exceeds available data (50)
  → Will use all 50 entries
```

**Solution**:

1. Increase number of source files
2. Adjust `--num-samples` to reasonable value
3. Check for duplicate files

---

## Advanced Usage

### Batch Convert All LongBench Datasets

```bash
#!/bin/bash
mkdir -p data/longbench

# All English QA datasets
qa_datasets=(
    "narrativeqa"
    "qasper"
    "multifieldqa_en"
    "hotpotqa"
    "2wikimqa"
    "musique"
)

# Batch conversion
for dataset in "${qa_datasets[@]}"; do
    echo "Converting: $dataset"
    python convert_longbench.py \
        --dataset "$dataset" \
        --num-samples 100 \
        --output "data/longbench/${dataset}.jsonl"
done

echo "Batch conversion complete!"
```

### Validate Output Format

```bash
# Validate JSONL format
python tests/test_jsonl_output.py data/output.jsonl

# Example output:
# Test file: data/output.jsonl
# ✅ JSONL format correct!
#    Total lines: 100
#    Valid objects: 100
#
# First object example:
#   Keys: ['text']
#   text preview: Once upon a time...
```

### Manual Format Conversion (If Needed)

**JSON Array → JSONL**:

```python
#!/usr/bin/env python3
import json

# Read JSON array
with open('old_format.json', 'r') as f:
    data = json.load(f)

# Write JSONL
with open('new_format.jsonl', 'w') as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
```

Or use command-line tool `jq`:

```bash
cat old_format.json | jq -c '.[]' > new_format.jsonl
```

---

## Dataset Statistics

### Average Length (tokens)

| Dataset | Average Length | Task Type |
|--------|----------|----------|
| narrativeqa | 18K | Long novel QA |
| qasper | 5K | Academic paper QA |
| gov_report | 9K | Government report summary |
| hotpotqa | 3K | Multi-document QA |
| multifieldqa | 4K | Multi-field QA |

### Recommended Configuration

Choose parameters based on dataset average length:

**Short Text (< 2K tokens)**:
```bash
--max-input-length 4096
--timeout 60
```

**Medium Length (2K - 8K tokens)**:
```bash
--max-input-length 8192
--timeout 300
```

**Long Text (> 8K tokens)**:
```bash
--max-input-length 16384
--timeout 600
```

---

## Complete Workflow

### End-to-End Example

```bash
# 1. Convert data
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 100 \
  --output data/narrativeqa.jsonl

# 2. Validate format
python tests/test_jsonl_output.py data/narrativeqa.jsonl

# 3. Small-scale test
python fluxperf.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 5 \
  --concurrency 1

# 4. Formal stress test
python fluxperf.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 20 \
  --max-input-length 8192 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate \
  --slo-file examples/slo_example.yaml \
  --output-dir results/narrativeqa

# 5. Analyze results
cat results/narrativeqa/metrics_summary.csv
```

---

## References

- [LongBench Paper](https://arxiv.org/abs/2308.14508)
- [LongBench GitHub](https://github.com/THUDM/LongBench)
- [LongBench HuggingFace](https://huggingface.co/datasets/THUDM/LongBench)
- [JSONL Format Specification](http://jsonlines.org/)
