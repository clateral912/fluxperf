# 多轮压测工作流说明

## 目录
- [数据准备](#数据准备)
- [本地 Mock 服务](#本地-mock-服务)
- [压测执行流程](#压测执行流程)
- [调试与日志](#调试与日志)
- [常见问题](#常见问题)

---

## 数据准备

### 1. ShareGPT 数据集处理

使用 `process_sharegpt.py` 提取 human 发言：

```bash
python process_sharegpt.py data/ShareGPT/ sharegpt_clean.jsonl --max-sessions 1000
```

输出 JSONL 每行示例：

```json
{"session_id": "session-1", "user_messages": ["问题1", "问题2"], "source": "data/ShareGPT/sg.json"}
```

### 2. LongBench 转换

```bash
python convert_longbench.py --dataset narrativeqa --num-samples 200 --output data/nq.jsonl
```

可与 ShareGPT 合并：

```bash
cat sharegpt_clean.jsonl data/nq.jsonl > data/mixed.jsonl
```

### 3. 数据格式要求

工具支持三种多轮格式：
1. `{"session_id": "s1", "user_messages": ["问1", "问2"]}`
2. `{"conversations": [{"from": "human", "value": "hi"}, ...]}`
3. `{"messages": [{"role": "user", "content": "问"}, ...]}`

---

## 本地 Mock 服务

启动内置 mock：

```bash
python dual_round_benchmarker.py \
  --dataset examples/example_dataset.json \
  --mock-server \
  --mock-host 127.0.0.1 \
  --mock-port 8765 \
  --num-samples 4 \
  --concurrency 2
```

独立启动：

```bash
python llm_mocker.py --host 0.0.0.0 --port 8001
```

健康检查：`curl http://127.0.0.1:8001/health`

---

## 压测执行流程

### 1. 基础命令

```bash
python dual_round_benchmarker.py \
  --dataset data/mixed.jsonl \
  --endpoint http://127.0.0.1:8001/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 \
  --max-context-tokens 4096 \
  --max-output-tokens 512 \
  --output-dir results/mixed
```

### 2. Session 历史逻辑

- 首次请求：历史为空，仅发送当前问
- 返回后追加 `{user, assistant}` 对
- 新请求前自动截断超出的最老轮次

### 3. KV Cache 重置

若 API 提供缓存清除接口，可使用：

```
--reset-cache-url http://127.0.0.1:8001/reset
--reset-cache-between-rounds
```

---

## 调试与日志

### 1. 请求保存

```
--save-requests --output-dir debug_requests
```
生成 `requests_round{n}_conc{m}_timestamp.jsonl`。

### 2. 调试模式

```
--debug --debug-log-dir debug_logs
```
输出包含：
- session_id、turn_index
- 消息列表
- context_tokens / history_truncated

### 3. Prometheus 集成

```
--prometheus-url http://localhost:3001/metrics \
--prometheus-metrics lmcache_hit_rate memory_usage_bytes
```

---

## 常见问题

1. **没有安装 aiohttp**：运行 `pip install -r requirements.txt`
2. **历史过长导致截断频繁**：提升 `--max-context-tokens` 或减少会话轮数
3. **Mock 服务端口占用**：调整 `--mock-port` 或提前关闭占用进程
4. **JSONL 格式错误**：确保每行独立 JSON 且无额外逗号
