# Dual Round Benchmarker 架构文档

## 项目概述

Dual Round Benchmarker 是一个用于测试 LLM API 服务性能的工具,特别适合评估缓存性能(如 KV 缓存、前缀缓存)。通过两轮相同请求的性能对比,可以量化缓存的加速效果。

## 核心组件

### 1. dual_round_benchmarker.py

主压测工具,提供以下核心功能:

#### 核心类

- **BenchmarkConfig**: 配置数据类,包含所有测试参数
- **BenchmarkRunner**: 测试执行器,负责组织和执行测试流程
- **OpenAIClient**: HTTP 客户端,负责与 API endpoint 通信
- **MetricsAnalyzer**: 指标分析器,计算和展示性能指标
- **PrometheusCollector**: Prometheus 指标收集器
- **ConversationHistory**: 多轮对话历史管理

#### 测试模式

1. **DUAL_ROUND 模式**: 两轮单次请求,每个数据集条目作为独立会话
2. **MULTI_TURN 模式**: 多轮对话模式,保持会话上下文

#### 关键流程

```
加载配置 → 加载数据集 → 采样 → 第1轮测试 → 第2轮测试 → 分析指标 → 输出结果
                                   ↓                ↓
                           收集 Prometheus      收集 Prometheus
```

### 2. llm_mocker.py

Mock LLM 服务器,用于测试和开发:

- 实现 OpenAI Chat Completions API 兼容接口
- 支持流式输出(SSE)
- 可配置响应延迟和内容
- 用于环境变量测试和集成测试

### 3. convert_longbench.py

LongBench 数据集转换工具:

- 从 HuggingFace 下载数据集
- 转换为 benchmarker 兼容的 JSONL 格式
- 支持批量转换和采样

### 4. process_sharegpt.py

ShareGPT 数据集处理工具:

- 清洗和标准化 ShareGPT 格式数据
- 提取多轮对话
- 生成 JSONL 格式输出

## 数据流

### 输入

1. **数据集**: JSON/JSONL 格式,支持多种结构
   - 单轮: `{"text": "..."}`
   - 多轮: `{"user_messages": [...], ...}`
   - ShareGPT: `{"conversations": [...]}`

2. **配置文件**
   - Recipe YAML: 多阶段测试配置
   - SLO YAML: 服务级别目标定义

### 输出

1. **终端输出**: 实时进度和格式化表格
2. **CSV 文件**: 汇总指标对比表
3. **JSON 文件**: 详细的请求级别数据
4. **JSONL 日志**: 请求详情(可选)
5. **调试日志**: JSON 格式调试信息(可选)

## 性能指标

### 基础指标

- **TTFT (Time to First Token)**: 首 token 延迟(ms)
- **ITL (Inter-Token Latency)**: token 间延迟(ms)
- **Latency**: 总延迟(ms)
- **Throughput**: 吞吐量(tokens/sec, requests/sec)
- **Goodput**: 满足 SLO 的有效吞吐量

### 统计值

每个指标计算:
- avg, p50, p90, p95, p99, min, max, stddev

### Prometheus 指标

支持自定义 Prometheus 指标收集,常用指标:
- `lmcache_hit_rate`: 缓存命中率
- `vllm_gpu_cache_usage_perc`: GPU 缓存使用率
- `vllm_num_requests_running`: 运行中的请求数

## Recipe 系统

Recipe 文件支持多阶段测试配置:

```yaml
global:
  dataset: path/to/dataset.jsonl
  endpoint: http://localhost:8000/v1/chat/completions
  mode: multi_turn
  
stages:
  - name: "Stage 1"
    concurrency_levels: [5, 10]
    num_samples: [20, 40]
    env:
      CUSTOM_VAR: "value"
```

### 环境变量注入

每个 stage 可以设置独立的环境变量,用于:
- 控制服务器行为
- A/B 测试不同配置
- 动态调整参数

## Prometheus 集成

### 数据抓取机制

在 `PrometheusCollector` 类(dual_round_benchmarker.py:399-462):

```python
class PrometheusCollector:
    async def collect_during_test(self, session, start_time, end_time, interval=1.0):
        while time.time() < end_time:
            metrics_data = await self.fetch_metrics(session)
            # 存储带时间戳的指标
            await asyncio.sleep(interval)
```

### 抓取频率

默认每 **0.5 秒** 抓取一次(dual_round_benchmarker.py:899):

```python
prom_task = asyncio.create_task(
    prom_collector.collect_during_test(
        client.session,
        round_start_time,
        estimated_end_time,
        interval=0.5  # 500ms 抓取间隔
    )
)
```

### 时间范围过滤

测试完成后,只保留测试时间范围内的数据点:

```python
def get_metrics_in_timerange(self, start_time: float, end_time: float):
    # 过滤 start_time <= timestamp <= end_time 的数据
```

## 缓存控制

### KVCache 重置

支持在以下时机重置 vLLM KVCache:

1. **轮次间重置**: `--reset-cache-between-rounds`
2. **并发层级间重置**: `--reset-cache-between-concurrency`
3. **测试结束**: 自动重置

重置通过 POST 请求到 `--reset-cache-url` 端点实现。

## 目录结构

```
dual_round_benchmark/
├── dual_round_benchmarker.py    # 主程序
├── llm_mocker.py                # Mock 服务器
├── convert_longbench.py         # 数据转换
├── process_sharegpt.py          # ShareGPT 处理
├── requirements.txt             # 依赖
├── README.md                    # 项目说明
├── examples/                    # 配置示例
│   ├── recipe_*.yaml
│   └── slo_example.yaml
├── datasets/                    # 数据集
├── docs/                        # 文档
│   ├── ARCHITECTURE.md         # 本文档
│   ├── benchmarker_guide.md    # 使用指南
│   ├── RECIPE_GUIDE.md         # Recipe 配置
│   └── ENV_VAR_TESTING.md      # 环境变量测试
├── tests/                       # 测试套件
└── benchmark_results/           # 输出目录(生成)
```

## 扩展点

### 添加新的数据集格式

在 `_extract_user_messages_from_entry()` 方法中添加解析逻辑。

### 添加新的性能指标

1. 在 `RequestMetrics` 中添加字段
2. 在 `OpenAIClient.send_completion_request()` 中计算
3. 在 `MetricsAnalyzer.analyze_round()` 中聚合
4. 在 `MetricsAnalyzer.print_metrics()` 中展示

### 集成新的监控系统

参考 `PrometheusCollector` 实现新的 Collector 类。
