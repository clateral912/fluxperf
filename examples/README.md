# Recipe 示例

此目录包含各种 Recipe 配置文件示例。

## 文件列表

### 1. `recipe_example.yaml`
基本的 multi-turn 模式示例，展示完整的配置选项。

**用途**: 学习 Recipe 配置的所有功能

**运行**:
```bash
python dual_round_benchmarker.py --recipe examples/recipe_example.yaml
```

### 2. `recipe_dual_round.yaml`
Dual-round 模式示例，适用于 LongBench 等单轮问答数据集。

**用途**: 测试缓存效果、两轮一致性

**运行**:
```bash
python dual_round_benchmarker.py --recipe examples/recipe_dual_round.yaml
```

### 3. `recipe_env_test.yaml`
环境变量测试 Recipe，用于验证环境变量管理功能。

**用途**: 测试不同环境配置（GPU、后端等）

**运行**:
```bash
python dual_round_benchmarker.py --recipe examples/recipe_env_test.yaml
```

## Recipe 配置要素

每个 Recipe 文件包含：

1. **global**: 全局配置
   - `dataset`: 数据集路径
   - `mode`: `dual_round` 或 `multi_turn`
   - `endpoint`: API 端点
   - `model`: 模型名称
   - 其他可选配置

2. **mock_server** (可选): Mock 服务器配置
   - `enabled`: 是否启用
   - `host`: 主机地址
   - `port`: 端口号

3. **stages**: 测试阶段列表
   - `name`: 阶段名称
   - `env`: 环境变量
   - `concurrency_levels`: 并发层级
   - `num_samples`: 样本数量

## 快速开始

### 使用 Mock Server 测试

```bash
# 使用内置 mock server，无需真实 LLM 服务
python dual_round_benchmarker.py --recipe examples/recipe_env_test.yaml
```

### 连接真实服务

修改 recipe 文件中的 `endpoint` 和 `mock_server.enabled`:

```yaml
global:
  endpoint: "http://your-llm-service:8001/v1/chat/completions"

mock_server:
  enabled: false  # 禁用 mock server
```

## 自定义 Recipe

1. 复制示例文件
2. 修改配置参数
3. 调整 stages 以匹配你的测试场景
4. 运行并查看结果

详细文档请参考: [RECIPE_GUIDE.md](../RECIPE_GUIDE.md)
