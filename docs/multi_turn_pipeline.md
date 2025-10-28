# Multi-Turn Load Testing Workflow Guide

## Table of Contents
- [Data Preparation](#data-preparation)
- [Local Mock Service](#local-mock-service)
- [Load Testing Execution](#load-testing-execution)
- [Debugging and Logging](#debugging-and-logging)
- [Common Issues](#common-issues)

---

## Data Preparation

### 1. ShareGPT Dataset Processing

Use `process_sharegpt.py` to extract human messages:

```bash
python process_sharegpt.py data/ShareGPT/ sharegpt_clean.jsonl --max-sessions 1000
```

Output JSONL line example:

```json
{"session_id": "session-1", "user_messages": ["question1", "question2"], "source": "data/ShareGPT/sg.json"}
```

### 2. LongBench Conversion

```bash
python convert_longbench.py --dataset narrativeqa --num-samples 200 --output data/nq.jsonl
```

Can be merged with ShareGPT:

```bash
cat sharegpt_clean.jsonl data/nq.jsonl > data/mixed.jsonl
```

### 3. Data Format Requirements

The tool supports three multi-turn formats:
1. `{"session_id": "s1", "user_messages": ["question1", "question2"]}`
2. `{"conversations": [{"from": "human", "value": "hi"}, ...]}`
3. `{"messages": [{"role": "user", "content": "question"}, ...]}`

---

## Local Mock Service

Start built-in mock:

```bash
python fluxperf.py \
  --dataset examples/example_dataset.json \
  --mock-server \
  --mock-host 127.0.0.1 \
  --mock-port 8765 \
  --num-samples 4 \
  --concurrency 2
```

Start independently:

```bash
python llm_mocker.py --host 0.0.0.0 --port 8001
```

Health check: `curl http://127.0.0.1:8001/health`

---

## Load Testing Execution

### 1. Basic Command

```bash
python fluxperf.py \
  --dataset data/mixed.jsonl \
  --endpoint http://127.0.0.1:8001/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 \
  --max-context-tokens 4096 \
  --max-output-tokens 512 \
  --output-dir results/mixed
```

### 2. Session History Logic

- First request: Empty history, only send current question
- After response: Append `{user, assistant}` pair
- Before new request: Automatically truncate oldest turns if exceeded

### 3. KV Cache Reset

If API provides cache clearing interface, use:

```
--reset-cache-url http://127.0.0.1:8001/reset
--reset-cache-between-rounds
```

---

## Debugging and Logging

### 1. Request Saving

```
--save-requests --output-dir debug_requests
```
Generates `requests_round{n}_conc{m}_timestamp.jsonl`.

### 2. Debug Mode

```
--debug --debug-log-dir debug_logs
```
Output includes:
- session_id, turn_index
- Message list
- context_tokens / history_truncated

### 3. Prometheus Integration

```
--prometheus-url http://localhost:3001/metrics \
--prometheus-metrics lmcache_hit_rate memory_usage_bytes
```

---

## Common Issues

1. **aiohttp not installed**: Run `pip install -r requirements.txt`
2. **History too long causing frequent truncation**: Increase `--max-context-tokens` or reduce session turns
3. **Mock service port occupied**: Adjust `--mock-port` or close occupying process
4. **JSONL format error**: Ensure each line is independent JSON with no extra commas
