# 快速开始指南

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd dual_round_benchmark

# 安装依赖
pip install -r requirements.txt
```

## 基础使用

### 1. 使用 Mock 服务器测试

最简单的方式,无需真实 LLM 服务:

```bash
python dual_round_benchmarker.py \
  --mock-server \
  --dataset examples/example_dataset.json \
  --num-samples 10 \
  --concurrency 5
```

### 2. 连接真实 API

```bash
python dual_round_benchmarker.py \
  --dataset datasets/MixedBench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10
```

### 3. 多并发测试

```bash
python dual_round_benchmarker.py \
  --dataset datasets/MixedBench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 20 40 80 \
  --concurrency 5 10 20
```

### 4. 启用 Prometheus 监控

```bash
python dual_round_benchmarker.py \
  --dataset datasets/MixedBench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate vllm_gpu_cache_usage_perc
```

### 5. 使用 SLO 约束

创建 `slo.yaml`:

```yaml
constraints:
  ttft_ms:
    max: 1000
  itl_ms:
    max: 50
  latency_ms:
    max: 10000
```

运行测试:

```bash
python dual_round_benchmarker.py \
  --dataset datasets/MixedBench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10 \
  --slo-file slo.yaml
```

### 6. 使用 Recipe 配置

创建 `my_recipe.yaml`:

```yaml
global:
  dataset: datasets/MixedBench.jsonl
  endpoint: http://localhost:8000/v1/chat/completions
  mode: multi_turn
  max_output_tokens: 256

stages:
  - name: "低并发测试"
    concurrency_levels: [5]
    num_samples: [20]
  
  - name: "高并发测试"
    concurrency_levels: [10, 20]
    num_samples: [40, 80]
```

运行:

```bash
python dual_round_benchmarker.py --recipe my_recipe.yaml
```

## 数据集准备

### 转换 LongBench 数据

```bash
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 100 \
  --output datasets/narrativeqa.jsonl
```

### 处理 ShareGPT 数据

```bash
python process_sharegpt.py \
  --input datasets/ShareGPT/sg_90k_part1.json \
  --output datasets/sharegpt_clean.jsonl \
  --max-turns 3
```

## 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集文件路径 | 必需 |
| `--endpoint` | API endpoint URL | 必需(或 `--mock-server`) |
| `--num-samples` | 每个并发层级的样本数 | 必需 |
| `--concurrency` | 并发数列表 | `[10]` |
| `--mode` | 测试模式: `dual_round` 或 `multi_turn` | `multi_turn` |
| `--max-output-tokens` | 最大输出 token 数 | 无限制 |
| `--prometheus-url` | Prometheus metrics 端点 | 无 |
| `--slo-file` | SLO 配置文件 | 无 |
| `--output-dir` | 结果输出目录 | `benchmark_results` |

## 输出说明

### 终端输出

实时显示进度条和格式化表格:

```
第1轮 (并发:10): 100%|████████| 50/50 [00:25<00:00]
✓ 完成,耗时: 25.43 秒

┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Statistic         ┃ avg    ┃ p99    ┃ p50    ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ TTFT (ms)         │ 245.32 │ 312.45 │ 238.15 │
│ ITL (ms)          │  18.25 │  25.33 │  17.82 │
│ Latency (ms)      │2456.78 │2901.12 │2401.23 │
└───────────────────┴────────┴────────┴────────┘
```

### 文件输出

- `benchmark_results/benchmark_YYYYMMDD_HHMMSS.csv`: CSV 格式汇总
- `benchmark_results/benchmark_YYYYMMDD_HHMMSS_params.txt`: 测试参数
- `benchmark_results.json`: JSON 格式详细数据(如果指定 `--output`)

## 下一步

- 详细使用说明: [benchmarker_guide.md](benchmarker_guide.md)
- Recipe 配置: [RECIPE_GUIDE.md](RECIPE_GUIDE.md)
- 架构文档: [ARCHITECTURE.md](ARCHITECTURE.md)
