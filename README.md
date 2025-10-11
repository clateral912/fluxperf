# Dual Round Benchmarker

一个用于测试 LLM API 服务性能的双轮压测工具，专门设计用于评估缓存性能和多轮请求的表现。

## 项目结构

```
dual_round_benchmark/
├── dual_round_benchmarker.py    # 主程序
├── llm_mocker.py                # Mock LLM 服务器
├── process_sharegpt.py          # ShareGPT 数据处理
├── convert_longbench.py         # LongBench 数据转换
├── requirements.txt             # Python 依赖
├── examples/                    # Recipe 配置示例
│   ├── README.md
│   ├── recipe_example.yaml      # 基础示例
│   ├── recipe_dual_round.yaml   # Dual-round 模式
│   └── recipe_env_test.yaml     # 环境变量测试
├── datasets/                    # 数据集目录
│   ├── README.md
│   ├── ShareGPT/               # ShareGPT 原始数据
│   ├── LongBench/              # LongBench 原始数据
│   ├── sharegpt_clean.jsonl    # 处理后的数据
│   └── MixedBench.jsonl        # 混合测试数据
├── docs/                        # 文档
│   ├── benchmarker_guide.md    # 使用指南
│   ├── RECIPE_GUIDE.md         # Recipe 配置指南
│   └── ENV_VAR_TESTING.md      # 环境变量测试
└── tests/                       # 测试文件
    ├── test_env_variables.py
    └── test_recipe_env_integration.sh
```

## 核心功能

### 1. 双轮压测 (`dual_round_benchmarker.py`)

对同一批请求进行两轮测试，用于评估 KV 缓存、前缀缓存等优化的效果。

**关键特性**：
- ✅ 支持多并发级别测试
- ✅ 流式输出 (SSE) 支持
- ✅ 详细的性能指标（TTFT, ITL, Latency, Throughput, Goodput）
- ✅ Prometheus 指标集成
- ✅ SLO 约束支持
- ✅ 多格式输出（命令行表格、CSV、JSON）

**快速开始**：
```bash
# 安装依赖
pip install -r requirements.txt

# 运行基础测试
python dual_round_benchmarker.py \
  --dataset examples/example_dataset.json \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 10 \
  --concurrency 5

# 运行完整测试（多并发 + Prometheus）
python dual_round_benchmarker.py \
  --dataset examples/example_dataset.json \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 20 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate memory_usage_bytes \
  --slo-file examples/slo_example.yaml \
  --output-dir results/test_run
```

### 2. LongBench 数据转换器 (`convert_longbench.py`)

将 LongBench 数据集转换为 benchmarker 可用的 JSONL 格式。

**关键特性**：
- ✅ 支持从 HuggingFace 自动下载
- ✅ 支持本地文件和目录批量转换
- ✅ 自动去重
- ✅ 智能格式推荐
- ✅ 采样保护（防止重复采样）

**快速开始**：
```bash
# 从 HuggingFace 下载并转换
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 100 \
  --output data/narrativeqa.jsonl

# 批量转换本地文件夹
python convert_longbench.py \
  --input-dir LongBench_data/ \
  --num-samples 200 \
  --output data/mixed.jsonl
```

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd dual_round_benchmark

# 安装依赖
pip install -r requirements.txt

# (可选) 如果需要使用 HuggingFace 下载数据
pip install datasets
```

## 目录结构

```
dual_round_benchmark/
├── dual_round_benchmarker.py    # 核心压测工具
├── convert_longbench.py         # 数据转换工具
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
├── docs/                        # 详细文档
│   ├── benchmarker_guide.md    # 压测工具完整指南
│   └── converter_guide.md      # 数据转换完整指南
├── examples/                    # 示例文件
│   ├── example_dataset.json    # 示例数据集
│   └── slo_example.yaml        # SLO 配置示例
├── tests/                       # 测试脚本
│   ├── test_jsonl_output.py    # JSONL 格式验证
│   └── test_prometheus.py      # Prometheus 集成测试
└── tools/                       # 辅助工具
    └── (空)
```

## 性能指标说明

### 基础指标

- **TTFT (Time to First Token)**: 首 token 延迟，从发送请求到收到第一个 token 的时间
- **ITL (Inter-Token Latency)**: Token 间延迟，生成连续 token 之间的平均间隔
- **Latency**: 总延迟，从发送请求到收到完整响应的时间
- **Throughput**: 吞吐量
  - Token 吞吐量：tokens/sec
  - Request 吞吐量：requests/sec
- **Goodput**: 满足 SLO 约束的有效吞吐量

### 统计值

每个指标都包含以下统计值：
- **Avg** (平均值)
- **P50** (中位数)
- **P90** (90 分位数)
- **P99** (99 分位数)
- **Min** (最小值)
- **Max** (最大值)
- **Stddev** (标准差)

### Prometheus 指标

支持查询任意 Prometheus 指标，常用指标：
- `lmcache_hit_rate`: 缓存命中率
- `memory_usage_bytes`: 内存使用
- `gpu_utilization`: GPU 利用率

## 数据集格式

### JSONL 格式（推荐）

每行一个 JSON 对象：
```
{"text": "第一个提示"}
{"text": "第二个提示"}
{"text": "第三个提示"}
```

### JSON 数组格式（也支持）

```json
[
  {"text": "第一个提示"},
  {"text": "第二个提示"}
]
```

## SLO 配置

创建 `slo.yaml` 文件定义服务级别目标：

```yaml
slo:
  ttft_ms: 1000        # TTFT < 1000ms
  itl_ms: 50           # ITL < 50ms
  latency_ms: 10000    # 总延迟 < 10000ms
```

只有同时满足所有 SLO 约束的请求才计入 Goodput。

## 输出结果

### 命令行输出

表格格式显示所有指标：

```
========================================================================================================================
并发: 5 | 第 1 轮性能指标
========================================================================================================================
总请求数: 50 | 成功: 50 | 失败: 0 | 测试时长: 25.43 秒
平均输入 tokens: 15.2 | 平均输出 tokens: 128.6
------------------------------------------------------------------------------------------------------------------------
│ Metric                       │ Avg          │ P50          │ P90          │ P99          │ Min          │ Max          │ Stddev       │
├──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ TTFT (ms)                    │ 245.32       │ 238.15       │ 298.45       │ 312.45       │ 210.12       │ 328.91       │ 25.67        │
│ ITL (ms)                     │ 18.25        │ 17.82        │ 23.12        │ 25.33        │ 12.34        │ 28.14        │ 3.45         │
│ Latency (ms)                 │ 2456.78      │ 2401.23      │ 2789.34      │ 2901.12      │ 2123.45      │ 3012.34      │ 198.76       │
└──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

Throughput:
  Token 吞吐量: 52.14 tokens/sec
  Request 吞吐量: 0.41 requests/sec

Goodput (SLO):
  满足 SLO 的请求数: 45 / 50 (90.00%)
```

### CSV 输出

自动生成 CSV 文件，便于在 Excel 中分析：
- `results/metrics_summary.csv`: 所有并发级别的汇总对比

### JSON 输出

完整的原始数据，便于程序化分析：
- `results/concurrency_5_round_1.json`: 详细的请求级别数据

## 使用场景

### 1. 缓存性能评估

通过对比两轮测试的性能差异，评估缓存效果：

```bash
python dual_round_benchmarker.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate
```

**预期结果**：
- 第 1 轮：缓存未命中，TTFT 较高
- 第 2 轮：缓存命中，TTFT 显著降低

### 2. 多并发压测

测试不同并发级别下的性能：

```bash
python dual_round_benchmarker.py \
  --dataset data/mixed.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 100 \
  --concurrency 1 5 10 20 50 100 \
  --output-dir results/scaling_test
```

### 3. SLO 合规性测试

验证服务是否满足 SLO 要求：

```bash
python dual_round_benchmarker.py \
  --dataset data/production_queries.jsonl \
  --endpoint http://production-api:8000/v1/chat/completions \
  --num-samples 1000 \
  --concurrency 50 \
  --slo-file slo_production.yaml \
  --output-dir results/slo_check
```

## 常见问题

### Q: 如何设置合适的并发数？

A: 建议从低到高逐步测试：`--concurrency 1 5 10 20 50`，观察性能拐点。

### Q: TTFT 和 ITL 的区别？

A:
- **TTFT**: 首次响应速度，影响用户感知延迟
- **ITL**: 持续生成速度，影响流式体验

### Q: Goodput 为 0% 怎么办？

A: 检查 SLO 配置是否过于严格，或者服务性能是否需要优化。

### Q: 支持非 OpenAI 格式的 API 吗？

A: 目前仅支持 OpenAI 兼容的 Chat Completions API。

## 详细文档

- **[Benchmarker 完整指南](docs/benchmarker_guide.md)**: 压测工具的详细使用说明
- **[Converter 完整指南](docs/converter_guide.md)**: 数据转换工具的详细说明
- **[开发者指南](CLAUDE.md)**: 代码架构和开发说明

## 示例和测试

### 示例文件
- `examples/example_dataset.json`: 10 条示例提示
- `examples/slo_example.yaml`: SLO 配置示例

### 测试工具
```bash
# 验证 JSONL 格式
python tests/test_jsonl_output.py data/output.jsonl

# 测试 Prometheus 集成
python tests/test_prometheus.py
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
