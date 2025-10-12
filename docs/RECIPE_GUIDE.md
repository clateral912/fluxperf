# Recipe 配置指南

## 概述

Recipe 功能允许你通过 YAML 配置文件定义复杂的多阶段压测场景，支持：

- **两种测试模式**: `dual_round` (单轮两次) 和 `multi_turn` (多轮对话)
- **多阶段测试**: 每个 stage 可以设置不同的并发参数和环境变量
- **环境变量管理**: 每个 stage 启动前设置环境变量，结束后自动恢复
- **统一配置**: 避免重复输入大量命令行参数

## 两种模式

### 1. Dual Round 模式 (`dual_round`)

适用于 **单轮问答数据集** (如 LongBench, MMLU 等):
- 每个数据条目只包含一个问题
- 第一轮: 发送所有问题
- 第二轮: 再次发送相同的问题 (可选打乱顺序)
- 用于测试缓存效果、一致性等

**示例数据格式**:
```jsonl
{"id": "1", "text": "What is the capital of France?"}
{"id": "2", "text": "Explain quantum computing"}
```

### 2. Multi Turn 模式 (`multi_turn`)

适用于 **多轮对话数据集** (如 ShareGPT, OpenAssistant 等):
- 每个数据条目包含多轮对话
- 第一轮: 按原始顺序发送所有对话
- 第二轮: 再次发送所有对话 (可选打乱会话顺序)
- 用于测试上下文管理、多轮一致性等

**示例数据格式**:
```jsonl
{
  "id": "conv_1",
  "conversations": [
    {"from": "human", "value": "Hello"},
    {"from": "gpt", "value": "Hi there!"},
    {"from": "human", "value": "How are you?"}
  ]
}
```

## Recipe 文件结构

```yaml
# 全局配置
global:
  dataset: "path/to/dataset.jsonl"
  endpoint: "http://localhost:8001/v1/chat/completions"
  model: "gpt-3.5-turbo"
  mode: "multi_turn"  # 或 "dual_round"
  
  # 可选配置
  timeout: 300
  max_output_tokens: 2048
  shuffle_round2: true
  output_dir: "results"

# Mock Server 配置 (可选)
mock_server:
  enabled: true
  host: "127.0.0.1"
  port: 8765

# 测试阶段
stages:
  - name: "Stage 1: Low Concurrency"
    env:
      CUDA_VISIBLE_DEVICES: "0"
      CUSTOM_VAR: "value1"
    concurrency_levels: [2, 4]
    num_samples: [4, 8]
  
  - name: "Stage 2: High Concurrency"
    env:
      CUDA_VISIBLE_DEVICES: "0,1"
      CUSTOM_VAR: "value2"
    concurrency_levels: [8, 16]
    num_samples: [16, 32]
```

## 使用方法

### 方式 1: 使用 Recipe 文件

```bash
python dual_round_benchmarker.py --recipe recipe_example.yaml
```

### 方式 2: 命令行参数

```bash
# Multi-turn 模式
python dual_round_benchmarker.py \
  --dataset sharegpt.jsonl \
  --endpoint http://localhost:8001/v1/chat/completions \
  --mode multi_turn \
  --num-samples 10 \
  --concurrency 5

# Dual-round 模式
python dual_round_benchmarker.py \
  --dataset longbench.jsonl \
  --endpoint http://localhost:8001/v1/chat/completions \
  --mode dual_round \
  --num-samples 20 \
  --concurrency 10
```

## 配置参数详解

### Global 配置

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `dataset` | string | 是 | 数据集文件路径 |
| `endpoint` | string | 条件* | API endpoint URL |
| `mode` | string | 是 | `dual_round` 或 `multi_turn` |
| `model` | string | 否 | 模型名称 (默认: gpt-3.5-turbo) |
| `timeout` | int | 否 | 请求超时时间/秒 (默认: 300) |
| `max_output_tokens` | int | 否 | 最大输出 token 数 |
| `max_context_tokens` | int | 否 | 最大上下文 token 数 (超出时截断历史) |
| `shuffle_round2` | bool | 否 | 第二轮是否打乱顺序 (默认: true) |
| `output_dir` | string | 否 | 结果输出目录 |
| `slo_file` | string | 否 | SLO 配置文件路径 |
| `prometheus_url` | string | 否 | Prometheus metrics URL |
| `save_requests` | bool | 否 | 是否保存请求日志 |
| `debug` | bool | 否 | 是否启用调试模式 |

*当 `mock_server.enabled=true` 时可省略

### Stage 配置

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | string | 否 | 阶段名称 |
| `concurrency_levels` | list[int] | 是 | 并发层级列表 |
| `num_samples` | list[int] | 是 | 每个并发层级的样本数 |
| `env` | dict | 否 | 环境变量键值对 |

## 示例 Recipe

### 示例 1: Multi-turn 模式 (ShareGPT)

```yaml
global:
  dataset: "datasets/sharegpt_clean.jsonl"
  endpoint: "http://localhost:8001/v1/chat/completions"
  mode: "multi_turn"
  model: "llama-3-8b"
  max_context_tokens: 4096

stages:
  - name: "Warm-up"
    concurrency_levels: [2]
    num_samples: [4]
  
  - name: "Production Load"
    env:
      VLLM_ATTENTION_BACKEND: "FLASH_ATTN"
    concurrency_levels: [10, 20, 40]
    num_samples: [20, 40, 80]
```

### 示例 2: Dual-round 模式 (LongBench)

```yaml
global:
  dataset: "longbench_qasper.jsonl"
  mode: "dual_round"
  model: "qwen-72b"
  shuffle_round2: true

mock_server:
  enabled: true
  port: 8765

stages:
  - name: "Cache Cold"
    env:
      ENABLE_PREFIX_CACHE: "false"
    concurrency_levels: [5, 10]
    num_samples: [10, 20]
  
  - name: "Cache Warm"
    env:
      ENABLE_PREFIX_CACHE: "true"
    concurrency_levels: [5, 10]
    num_samples: [10, 20]
```

## 环境变量管理

每个 stage 可以设置专属的环境变量。运行流程:

1. **Stage 开始前**: 保存当前环境变量，设置 stage 指定的新值
2. **Stage 运行**: 测试过程中使用新的环境变量
3. **Stage 结束后**: 恢复原始环境变量

这允许你在不同 stage 测试不同配置，例如:
- 不同的 GPU 设备 (`CUDA_VISIBLE_DEVICES`)
- 不同的后端 (`VLLM_ATTENTION_BACKEND`)
- 不同的缓存策略
- 自定义应用配置

## 最佳实践

1. **从小规模开始**: 第一个 stage 使用较小的并发和样本数
2. **逐步增加负载**: 后续 stage 逐渐提高并发
3. **合理设置样本数**: 建议 `num_samples >= 2 * concurrency`
4. **使用有意义的名称**: Stage 名称应清晰描述测试目的
5. **隔离环境变量影响**: 每个 stage 只设置必要的环境变量
6. **启用 Mock Server 进行测试**: 在实际测试前先用 mock 验证配置

## 故障排查

### Recipe 加载失败
- 检查 YAML 语法是否正确
- 确保所有必需字段都已填写
- 验证文件路径是否存在

### 环境变量未生效
- 确认应用程序会读取这些环境变量
- 检查是否需要重启服务才能应用变量

### 连接失败
- 确认 endpoint URL 正确
- 检查服务是否正在运行
- 如果使用 mock server，确保 `enabled: true`

## 进阶用法

### 组合多个 Recipe

你可以创建多个 Recipe 文件，分别测试不同场景:

```bash
# 测试缓存效果
python dual_round_benchmarker.py --recipe recipes/cache_test.yaml

# 测试长上下文
python dual_round_benchmarker.py --recipe recipes/long_context.yaml

# 测试高并发
python dual_round_benchmarker.py --recipe recipes/stress_test.yaml
```

### 与 CI/CD 集成

Recipe 文件可以版本控制，在 CI/CD 中自动运行:

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmark
  run: |
    python dual_round_benchmarker.py --recipe ci_recipe.yaml
```
