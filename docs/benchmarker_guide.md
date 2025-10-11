# Dual Round Benchmarker 完整指南

## 目录
- [概述](#概述)
- [安装和配置](#安装和配置)
- [核心功能](#核心功能)
- [性能指标](#性能指标)
- [命令行参数](#命令行参数)
- [输出格式](#输出格式)
- [使用示例](#使用示例)
- [SLO 配置](#slo-配置)
- [Prometheus 集成](#prometheus-集成)
- [故障排除](#故障排除)

---

## 概述

`dual_round_benchmarker.py` 是一个专门设计用于测试 LLM API 服务性能的工具，特别适合评估缓存性能（如 KV 缓存、前缀缓存）。

### 核心特性

- **双轮测试**: 对同一批请求执行两次，用于对比缓存前后的性能差异
- **多并发支持**: 可同时测试多个并发级别（如 5, 10, 20, 50）
- **流式输出**: 支持 Server-Sent Events (SSE) 流式响应
- **丰富的指标**: TTFT, ITL, Latency, Throughput, Goodput + Prometheus 自定义指标
- **多种输出**: 命令行表格、CSV、JSON 三种格式
- **SLO 约束**: 基于服务级别目标计算 Goodput

### 双轮测试原理

**第 1 轮**：冷启动，缓存未命中
```
请求 1 → API (无缓存) → 响应 1  ← TTFT 较高
请求 2 → API (无缓存) → 响应 2
...
```

**第 2 轮**：相同请求，缓存命中
```
请求 1 → API (缓存命中) → 响应 1  ← TTFT 降低
请求 2 → API (缓存命中) → 响应 2
...
```

通过对比两轮的性能指标，可以量化缓存的加速效果。

---

## 安装和配置

### 系统要求

- Python 3.8+
- 支持 OpenAI Chat Completions API 格式的 LLM 服务

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖包：
- `aiohttp>=3.8.0` - 异步 HTTP 客户端
- `pyyaml>=6.0` - YAML 配置解析
- `tqdm>=4.65.0` - 进度条显示
- `prometheus-client>=0.16.0` - Prometheus 指标查询

---

## 核心功能

### 1. 性能指标测量

#### TTFT (Time to First Token)
首 token 延迟，从发送请求到收到第一个 token 的时间。

**影响因素**：
- 模型加载时间
- Prompt 处理时间
- KV 缓存命中率

#### ITL (Inter-Token Latency)
Token 间延迟，生成连续 token 之间的平均时间间隔。

**计算方式**：
```
ITL = (总生成时间 - TTFT) / (生成的 token 数 - 1)
```

#### Latency (总延迟)
从发送请求到收到完整响应的总时间。

```
Latency = TTFT + (ITL × 生成的 token 数)
```

### 2. 吞吐量 (Throughput)

- **Token 吞吐量**: 每秒生成的 token 数
- **Request 吞吐量**: 每秒完成的请求数

```
Token Throughput = 总生成 tokens / 总时长
Request Throughput = 总请求数 / 总时长
```

### 3. Goodput (有效吞吐量)

满足 SLO 约束的有效吞吐量。

```
Goodput = (满足 SLO 的 tokens) / 总时长
```

### 4. Prometheus 指标集成

支持查询任意 Prometheus 指标，常用指标：

- `lmcache_hit_rate`: 缓存命中率
- `memory_usage_bytes`: 内存使用
- `gpu_utilization`: GPU 利用率
- `kv_cache_size`: KV 缓存大小

---

## 性能指标

每个指标都包含以下统计值：

| 统计值 | 说明 | 用途 |
|--------|------|------|
| **Avg** | 平均值 | 整体性能水平 |
| **P50** | 中位数 | 典型用户体验 |
| **P90** | 90分位数 | 大部分用户体验 |
| **P99** | 99分位数 | 尾部延迟 |
| **Min** | 最小值 | 最佳性能 |
| **Max** | 最大值 | 最差性能 |
| **Stddev** | 标准差 | 性能稳定性 |

---

## 命令行参数

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset` | 数据集文件路径 | `--dataset data/test.jsonl` |
| `--endpoint` | API 端点 URL | `--endpoint http://localhost:8000/v1/chat/completions` |

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-samples` | 全部 | 测试的样本数量 |
| `--concurrency` | 1 | 并发数（可指定多个） |
| `--model` | 自动 | 模型名称 |
| `--timeout` | 300 | 请求超时（秒） |

### 可选参数

| 参数 | 说明 |
|------|------|
| `--max-input-length` | 截断输入长度 |
| `--max-output-tokens` | 限制最大生成 tokens |
| `--temperature` | 生成温度 |
| `--top-p` | Top-p 采样 |
| `--output-dir` | 输出目录 |
| `--slo-file` | SLO 配置文件 |

### Prometheus 参数

| 参数 | 说明 |
|------|------|
| `--prometheus-url` | Prometheus 指标端点 |
| `--prometheus-metrics` | 要查询的指标列表 |

### 完整示例

```bash
python dual_round_benchmarker.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model meta-llama/Llama-3-8B \
  --num-samples 100 \
  --concurrency 5 10 20 \
  --max-input-length 8192 \
  --max-output-tokens 512 \
  --temperature 0.7 \
  --timeout 600 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate memory_usage_bytes \
  --slo-file slo.yaml \
  --output-dir results/test_run
```

---

## 输出格式

### 1. 命令行表格输出

```
========================================================================================================================
并发: 10 | 第 1 轮性能指标
========================================================================================================================
总请求数: 100 | 成功: 100 | 失败: 0 | 测试时长: 45.32 秒
平均输入 tokens: 1024.5 | 平均输出 tokens: 256.3
------------------------------------------------------------------------------------------------------------------------
│ Metric                       │ Avg          │ P50          │ P90          │ P99          │ Min          │ Max          │ Stddev       │
├──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ TTFT (ms)                    │ 245.32       │ 238.15       │ 298.45       │ 312.45       │ 210.12       │ 328.91       │ 25.67        │
│ ITL (ms)                     │ 18.25        │ 17.82        │ 23.12        │ 25.33        │ 12.34        │ 28.14        │ 3.45         │
│ Latency (ms)                 │ 4756.78      │ 4501.23      │ 5789.34      │ 6201.12      │ 4023.45      │ 6512.34      │ 398.76       │
│ lmcache_hit_rate             │ 0.0000       │ 0.0000       │ 0.0000       │ 0.0000       │ 0.0000       │ 0.0000       │ 0.0000       │
└──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

Throughput:
  Token 吞吐量: 565.14 tokens/sec
  Request 吞吐量: 2.21 requests/sec

Goodput (SLO):
  满足 SLO 的请求数: 95 / 100 (95.00%)
  满足 SLO 的 tokens: 24348 (95.12%)
```

### 2. CSV 输出

文件: `results/metrics_summary.csv`

```csv
指标/轮次,并发5-第1轮,并发5-第2轮,并发10-第1轮,并发10-第2轮

TTFT (ms)
平均值,245.32,98.45,312.54,105.12
中位数,238.15,95.20,305.23,102.11
P90,289.45,145.67,378.12,145.23
P99,312.45,156.30,398.45,156.78
最小值,210.12,80.23,280.34,85.45
最大值,350.67,190.34,450.12,210.23
标准差,25.67,18.34,35.67,20.45

ITL (ms)
平均值,18.25,16.34,20.45,17.67
...
```

### 3. JSON 输出

文件: `results/concurrency_10_round_1.json`

```json
{
  "configuration": {
    "endpoint": "http://localhost:8000/v1/chat/completions",
    "concurrency": 10,
    "num_samples": 100,
    "round": 1
  },
  "summary": {
    "total_requests": 100,
    "successful_requests": 100,
    "failed_requests": 0,
    "avg_ttft": 245.32,
    "p50_ttft": 238.15,
    ...
  },
  "requests": [
    {
      "index": 0,
      "prompt": "...",
      "ttft": 235.67,
      "itl": 18.45,
      "latency": 4523.12,
      "input_tokens": 1024,
      "output_tokens": 256,
      "success": true
    },
    ...
  ]
}
```

---

## 使用示例

### 示例 1: 基础测试

```bash
python dual_round_benchmarker.py \
  --dataset examples/example_dataset.json \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 10 \
  --concurrency 5
```

### 示例 2: 缓存性能评估

```bash
python dual_round_benchmarker.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate \
  --output-dir results/cache_test
```

**预期结果**：
- 第 1 轮: `lmcache_hit_rate ≈ 0.0`, TTFT 高
- 第 2 轮: `lmcache_hit_rate > 0.8`, TTFT 显著降低

### 示例 3: 多并发压测

```bash
python dual_round_benchmarker.py \
  --dataset data/mixed.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 200 \
  --concurrency 1 5 10 20 50 100 \
  --output-dir results/scaling_test
```

查看 `results/scaling_test/metrics_summary.csv` 对比不同并发下的性能。

### 示例 4: SLO 合规性测试

```bash
python dual_round_benchmarker.py \
  --dataset data/production_queries.jsonl \
  --endpoint http://production:8000/v1/chat/completions \
  --num-samples 1000 \
  --concurrency 50 \
  --slo-file slo_production.yaml \
  --output-dir results/slo_check
```

---

## SLO 配置

创建 `slo.yaml` 文件定义服务级别目标：

```yaml
slo:
  ttft_ms: 1000        # TTFT 必须 < 1000ms
  itl_ms: 50           # ITL 必须 < 50ms
  latency_ms: 10000    # 总延迟必须 < 10000ms
```

**Goodput 计算规则**：

只有同时满足所有 SLO 约束的请求才计入 Goodput：

```python
if (ttft < slo.ttft_ms) AND (itl < slo.itl_ms) AND (latency < slo.latency_ms):
    goodput_tokens += output_tokens
    goodput_requests += 1
```

**示例输出**：

```
Goodput (SLO):
  满足 SLO 的请求数: 850 / 1000 (85.00%)
  满足 SLO 的 tokens: 217600 / 256000 (85.00%)
```

---

## Prometheus 集成

### 配置

```bash
python dual_round_benchmarker.py \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate memory_usage_bytes gpu_utilization \
  ...
```

### 指标查询

工具会在测试前后查询 Prometheus 指标，并计算差值或平均值。

**查询时机**：
1. 每个请求发送前：记录 `metric_before`
2. 每个请求完成后：记录 `metric_after`
3. 计算：`metric_delta = metric_after - metric_before`

### 常用指标

| 指标名称 | 说明 | 典型值 |
|----------|------|--------|
| `lmcache_hit_rate` | 缓存命中率 | 0.0 - 1.0 |
| `memory_usage_bytes` | 内存使用（字节） | 1GB - 100GB |
| `gpu_utilization` | GPU 利用率 | 0.0 - 1.0 |
| `kv_cache_size` | KV 缓存大小 | - |
| `request_latency` | 请求延迟 | - |

### 输出示例

```
Prometheus 指标

lmcache_hit_rate
平均值  0.0000   0.8521   0.0012   0.9234
P50     0.0000   0.8500   0.0000   0.9200
P90     0.0000   0.9100   0.0050   0.9500
P99     0.0000   0.9300   0.0100   0.9700
最小值  0.0000   0.7800   0.0000   0.8900
最大值  0.0000   0.9500   0.0200   0.9800
标准差  0.0000   0.0420   0.0045   0.0280
```

---

## 故障排除

### 问题 1: 连接超时

**错误信息**: `asyncio.TimeoutError` 或 `Connection timeout`

**解决方案**:
```bash
# 增加超时时间
python dual_round_benchmarker.py \
  --timeout 600 \
  ...
```

### 问题 2: 所有请求失败

**错误信息**: `成功: 0 | 失败: 100`

**检查清单**:
1. API 端点是否正确
2. 服务是否启动
3. 模型名称是否正确
4. 查看 JSON 输出的错误详情

### 问题 3: Goodput 为 0%

**原因**: SLO 配置过于严格

**解决方案**:
1. 查看实际的性能指标
2. 调整 SLO 配置为合理值
3. 或者优化服务性能

### 问题 4: TTFT 异常高

**可能原因**:
- 模型未加载到显存
- 输入过长
- 服务器负载过高

**检查**:
```bash
# 查看 Prometheus 指标
--prometheus-metrics gpu_memory_usage model_load_time
```

### 问题 5: 第 2 轮没有加速

**可能原因**:
- 缓存未启用
- 请求格式不完全一致
- 缓存容量不足被驱逐

**检查**:
```bash
# 查看缓存命中率
--prometheus-metrics lmcache_hit_rate kv_cache_eviction_count
```

---

## 高级用法

### 1. 自定义数据集

创建 JSONL 文件：

```
{"text": "写一首关于春天的诗"}
{"text": "解释量子纠缠的原理"}
{"text": "推荐三本科幻小说"}
```

### 2. 长文本测试

```bash
# 使用 LongBench 数据集
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 50 \
  --output data/long_context.jsonl

python dual_round_benchmarker.py \
  --dataset data/long_context.jsonl \
  --max-input-length 16384 \
  --endpoint http://localhost:8000/v1/chat/completions
```

### 3. 批量测试脚本

```bash
#!/bin/bash
CONCURRENCIES="1 5 10 20 50 100"

for conc in $CONCURRENCIES; do
    echo "Testing concurrency: $conc"
    python dual_round_benchmarker.py \
      --dataset data/test.jsonl \
      --endpoint http://localhost:8000/v1/chat/completions \
      --num-samples 100 \
      --concurrency $conc \
      --output-dir results/conc_$conc
done
```

---

## 性能优化建议

### 1. 并发数选择

- **低并发 (1-5)**: 测试单请求性能
- **中并发 (10-20)**: 常见生产场景
- **高并发 (50-100)**: 压力测试

### 2. 样本数选择

- **快速测试**: 10-50 样本
- **常规测试**: 100-500 样本
- **生产验证**: 1000+ 样本

### 3. 超时设置

根据输出长度调整：

```
timeout = TTFT_expected + (max_tokens × ITL_expected) + 缓冲时间
```

示例：
- 短输出 (< 100 tokens): `--timeout 60`
- 中等输出 (100-500 tokens): `--timeout 300`
- 长输出 (> 500 tokens): `--timeout 600`

---

## 输出文件说明

测试完成后，`--output-dir` 目录包含：

```
results/
├── metrics_summary.csv              # 所有并发的汇总对比
├── concurrency_5_round_1.json       # 并发5第1轮详细数据
├── concurrency_5_round_2.json       # 并发5第2轮详细数据
├── concurrency_10_round_1.json      # 并发10第1轮详细数据
└── concurrency_10_round_2.json      # 并发10第2轮详细数据
```

### CSV 文件结构

横向对比所有测试配置：

| 指标/轮次 | 并发5-第1轮 | 并发5-第2轮 | 并发10-第1轮 | 并发10-第2轮 |
|-----------|-------------|-------------|--------------|--------------|
| TTFT-平均值 | 245.32 | 98.45 | 312.54 | 105.12 |
| TTFT-P99 | 312.45 | 156.30 | 398.45 | 156.78 |
| ... | ... | ... | ... | ... |

### JSON 文件结构

包含请求级别的详细数据，便于程序化分析。

---

## 相关工具

- **convert_longbench.py**: 数据集转换工具
- **test_jsonl_output.py**: JSONL 格式验证
- **test_prometheus.py**: Prometheus 连接测试

---

## 参考资料

- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [Prometheus 文档](https://prometheus.io/docs/)
- [LongBench 数据集](https://github.com/THUDM/LongBench)
