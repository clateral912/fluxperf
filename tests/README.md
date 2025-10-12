# 测试套件

完整的单元测试套件，覆盖项目的核心功能。

## 测试文件

### 核心测试

| 文件 | 覆盖内容 | 测试数量 |
|------|----------|----------|
| `test_dataclasses.py` | 所有 dataclass 的创建和验证 | 8 |
| `test_utils.py` | 工具函数和辅助方法 | 6 |
| `test_benchmark_runner.py` | BenchmarkRunner 核心逻辑 | 8 |
| `test_llm_client.py` | LLMClient 和请求处理 | 4 |
| `test_env_variables.py` | 环境变量管理 | 2 |

### 集成测试

| 文件 | 覆盖内容 |
|------|----------|
| `test_llm_mocker.py` | Mock LLM 服务器功能 |
| `test_process_sharegpt.py` | ShareGPT 数据处理 |
| `test_conversation_history.py` | 对话历史管理 |
| `test_prometheus.py` | Prometheus 集成 |
| `test_jsonl_output.py` | JSONL 输出格式 |
| `test_recipe_env_integration.sh` | Recipe 环境变量集成测试 |

## 运行测试

### 运行所有测试

```bash
# 简单模式
bash tests/run_all_tests.sh

# 或逐个运行
python tests/test_dataclasses.py
python tests/test_utils.py
python tests/test_benchmark_runner.py
python tests/test_llm_client.py
python tests/test_env_variables.py
```

### 生成覆盖率报告

```bash
# 需要先安装 coverage
pip install coverage

# 运行覆盖率分析
bash tests/coverage_report.sh

# 查看 HTML 报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 测试覆盖范围

### test_dataclasses.py

测试所有数据类的创建、默认值和字段验证：

- ✓ `SLOConstraints` - 空值和带值创建
- ✓ `BenchmarkConfig` - 所有配置参数
- ✓ `RequestMetrics` - 请求指标记录
- ✓ `RoundMetrics` - 轮次统计信息
- ✓ `SessionData` - 会话数据管理
- ✓ `RecipeStage` - Recipe 阶段配置
- ✓ `Recipe` - Recipe 完整结构
- ✓ `BenchmarkMode` - 枚举类型

**覆盖**: 100% dataclass 定义

### test_utils.py

测试工具函数和辅助方法：

- ✓ `count_tokens()` - Token 计数逻辑
  - 正常文本
  - 空文本和 None
  - 多空格处理
  - 特殊字符

- ✓ `MetricsAnalyzer.calculate_percentile()` - 百分位数计算
  - P50, P90, P99
  - 空列表和单值

- ✓ `MetricsAnalyzer._truncate_text()` - 文本截断
  - 短文本、长文本
  - 边界情况

- ✓ `DatasetLoader.sample_entries()` - 数据集采样
  - 正常采样
  - 超出数据集大小
  - 空数据集

- ✓ `SLOLoader.validate_slo()` - SLO 验证
  - TTFT 超标
  - ITL 超标
  - Latency 超标
  - Throughput 不足
  - 错误请求

- ✓ `RecipeLoader.load_recipe()` - Recipe 加载
  - 有效 recipe
  - 无效模式

**覆盖**: 主要工具函数 90%+

### test_benchmark_runner.py

测试 BenchmarkRunner 的核心逻辑：

- ✓ `_sanitize_user_message()` - 消息清理
  - 正常文本、空文本
  - 字典格式
  - 去除空格

- ✓ `_extract_text()` - 文本提取
  - 字符串、字典多种格式
  - text, prompt, content, messages 字段
  - 类型转换

- ✓ `_total_turns()` - 轮次计算
  - 多个 session
  - 空列表、单 session

- ✓ `_reset_conversation_state()` - 状态重置
  - 清空历史
  - 保留用户消息

- ✓ `_entries_to_single_turn_sessions()` - Dual-round 模式转换
  - 不同格式的 entry
  - ID 处理
  - 空条目过滤

- ✓ `_normalize_sessions()` - Multi-turn 模式转换
  - ShareGPT 格式
  - ID 去重处理

- ✓ `_build_conversation_history()` - 对话历史构建
  - 不同 turn 的历史
  - 正确的消息顺序

- ✓ 对话历史截断
  - max_context_tokens 限制
  - 保留最近消息

**覆盖**: BenchmarkRunner 主要方法 85%+

### test_llm_client.py

测试 LLMClient 的功能：

- ✓ 客户端初始化
  - 配置正确
  - 初始状态

- ✓ 生命周期管理
  - initialize()
  - cleanup()

- ✓ Payload 构建
  - model_name
  - max_output_tokens
  - save_requests 模式

- ✓ Metrics 初始化
  - 所有字段正确
  - 默认值

**覆盖**: LLMClient 核心功能 75%+

### test_env_variables.py

测试环境变量管理：

- ✓ 环境变量设置和恢复
  - 新值设置
  - 原值恢复
  - 多 stage 隔离

- ✓ 嵌套环境变量
  - 多层嵌套
  - 正确恢复

**覆盖**: 环境变量管理逻辑 100%

## 测试最佳实践

### 1. 测试命名

```python
def test_function_name():
    """简洁描述测试内容"""
    print("测试 function_name...")
    # 测试代码
    print("✓ function_name 测试通过")
```

### 2. 断言使用

```python
# 使用清晰的断言
assert result == expected, f"期望 {expected}, 实际 {result}"

# 测试异常
try:
    func_that_should_fail()
    assert False, "应该抛出异常"
except ExpectedError:
    pass  # 测试通过
```

### 3. 测试隔离

每个测试应该：
- 独立运行
- 不依赖其他测试
- 清理自己的状态

### 4. 测试覆盖目标

- **核心功能**: 100% 覆盖
- **工具函数**: 90%+ 覆盖
- **边界情况**: 重点测试
- **错误处理**: 确保测试

## 添加新测试

1. **创建测试文件**: `tests/test_new_feature.py`

2. **编写测试**:
```python
#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_round_benchmarker import YourClass

def test_your_feature():
    """测试描述"""
    print("测试 your_feature...")
    # 测试代码
    assert condition
    print("✓ your_feature 测试通过")

if __name__ == '__main__':
    try:
        test_your_feature()
        print("\n所有测试通过! ✓")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
```

3. **添加到测试运行器**: 编辑 `run_all_tests.sh`

4. **运行测试**: `bash tests/run_all_tests.sh`

## CI/CD 集成

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install coverage
      - name: Run tests
        run: bash tests/run_all_tests.sh
      - name: Generate coverage
        run: bash tests/coverage_report.sh
```

## 故障排查

### 测试失败

1. 查看错误消息和堆栈跟踪
2. 检查是否有依赖缺失
3. 确认测试环境正确

### 导入错误

```bash
# 确保在项目根目录运行
cd /path/to/dual_round_benchmark
python tests/test_xxx.py
```

### 覆盖率低

1. 运行覆盖率报告查看未覆盖代码
2. 为关键路径添加测试
3. 测试错误处理分支

## 贡献指南

提交新功能时请：

1. ✓ 编写相应的单元测试
2. ✓ 确保所有测试通过
3. ✓ 保持测试覆盖率 > 80%
4. ✓ 更新测试文档
