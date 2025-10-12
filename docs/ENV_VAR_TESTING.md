# 环境变量测试指南

## 概述

Recipe 功能支持为每个 stage 设置独立的环境变量，并在 stage 结束后自动恢复原始值。

## 测试方法

### 1. 单元测试

运行环境变量逻辑的单元测试：

```bash
python tests/test_env_variables.py
```

**测试内容**:
- ✓ 环境变量在 stage 开始前正确设置
- ✓ 环境变量在 stage 结束后正确恢复
- ✓ 多个 stage 的环境变量互不干扰
- ✓ 嵌套环境变量场景

### 2. 集成测试

使用真实的 recipe 文件测试环境变量管理：

```bash
# 方法 1: 使用测试脚本
bash tests/test_recipe_env_integration.sh

# 方法 2: 手动运行 recipe
python dual_round_benchmarker.py --recipe recipe_env_test.yaml
```

在运行过程中，你可以在另一个终端监控环境变量：

```bash
# 启动 recipe 后，在另一个终端运行
watch -n 1 'ps aux | grep dual_round_benchmarker | head -1'
```

### 3. 查看 Stage 名称

现在 CLI 输出的表格标题会显示 stage 名称而不是 concurrency：

**使用 Recipe 时的输出**:
```
            Test Stage 1 - GPU 0 - Round 1
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳...
```

**不使用 Recipe 时的输出**:
```
   Dual Round Benchmarker | LLM Metrics (Concurrency: 2, Round: 1)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳...
```

## 测试 Recipe 示例

`recipe_env_test.yaml` 包含 3 个 stage，每个设置不同的环境变量：

```yaml
stages:
  - name: "Test Stage 1 - GPU 0"
    env:
      CUDA_VISIBLE_DEVICES: "0"
      TEST_STAGE_NAME: "stage_1"
      VLLM_ATTENTION_BACKEND: "FLASH_ATTN"
    
  - name: "Test Stage 2 - GPU 0,1"
    env:
      CUDA_VISIBLE_DEVICES: "0,1"
      TEST_STAGE_NAME: "stage_2"
      VLLM_ATTENTION_BACKEND: "XFORMERS"
  
  - name: "Test Stage 3 - All GPUs"
    env:
      CUDA_VISIBLE_DEVICES: "0,1,2,3"
      TEST_STAGE_NAME: "stage_3"
```

## 验证环境变量生效

### 方法 1: 在代码中检查

修改你的应用代码来打印环境变量：

```python
import os
print(f"当前 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
```

### 方法 2: 使用 wrapper 脚本

创建一个包装脚本来捕获环境变量：

```bash
#!/bin/bash
# save_env.sh
echo "Stage 环境变量:" >> /tmp/stage_env.log
env | grep -E "CUDA|TEST_|VLLM" >> /tmp/stage_env.log
echo "---" >> /tmp/stage_env.log

# 调用实际程序
exec "$@"
```

然后在你的服务中使用这个 wrapper。

### 方法 3: 查看进程环境变量

```bash
# 找到进程 PID
ps aux | grep your_service

# 查看环境变量
cat /proc/<PID>/environ | tr '\0' '\n'
```

## 常见用例

### 1. 测试不同的 GPU 配置

```yaml
stages:
  - name: "Single GPU"
    env:
      CUDA_VISIBLE_DEVICES: "0"
    concurrency_levels: [10]
    num_samples: [20]
  
  - name: "Multi GPU"
    env:
      CUDA_VISIBLE_DEVICES: "0,1,2,3"
    concurrency_levels: [40]
    num_samples: [80]
```

### 2. 测试不同的后端

```yaml
stages:
  - name: "FlashAttention"
    env:
      VLLM_ATTENTION_BACKEND: "FLASH_ATTN"
    concurrency_levels: [10]
    num_samples: [20]
  
  - name: "xFormers"
    env:
      VLLM_ATTENTION_BACKEND: "XFORMERS"
    concurrency_levels: [10]
    num_samples: [20]
```

### 3. 测试缓存开关

```yaml
stages:
  - name: "Cache Disabled"
    env:
      ENABLE_PREFIX_CACHE: "false"
    concurrency_levels: [10]
    num_samples: [20]
  
  - name: "Cache Enabled"
    env:
      ENABLE_PREFIX_CACHE: "true"
    concurrency_levels: [10]
    num_samples: [20]
```

## 故障排查

### 环境变量未生效

1. **检查应用是否读取环境变量**: 确认你的应用确实从环境变量读取配置
2. **检查是否需要重启**: 某些应用只在启动时读取环境变量
3. **检查 shell 展开**: 环境变量值会被转换为字符串，特殊字符需要转义

### 环境变量未恢复

1. **检查异常处理**: 如果 stage 中途失败，finally 块仍会恢复环境变量
2. **检查子进程**: 子进程继承的环境变量不会影响父进程

### Stage 名称未显示

确保你使用的是 `--recipe` 参数而不是命令行参数。只有通过 recipe 运行时才会显示 stage 名称。

## 最佳实践

1. **使用描述性名称**: Stage 名称应清楚说明测试目的
2. **最小化变量数量**: 只设置需要改变的环境变量
3. **文档化副作用**: 在 recipe 注释中说明环境变量的影响
4. **测试恢复**: 在测试 recipe 中添加验证环节确保变量正确恢复

## 调试技巧

### 打印所有环境变量

在 `run_recipe_benchmark` 函数的 stage 循环中添加：

```python
print("\n当前环境变量:")
for key in sorted(os.environ.keys()):
    if any(k in key for k in ['CUDA', 'VLLM', 'TEST']):
        print(f"  {key} = {os.environ[key]}")
```

### 保存环境变量快照

```python
import json

# Stage 开始前
before = dict(os.environ)

# Stage 结束后
after = dict(os.environ)

# 比较
diff = {k: (before.get(k), after.get(k)) 
        for k in set(before) | set(after) 
        if before.get(k) != after.get(k)}

print(json.dumps(diff, indent=2))
```
