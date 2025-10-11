#!/bin/bash

# KVCache 重置功能使用示例

# 示例 1: 仅在轮次之间重置 KVCache
# 适用场景: 测试每个并发层级内缓存对第二轮的影响，但不同并发层级间保留缓存
python dual_round_benchmarker.py \
  --dataset data/test.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 20 \
  --reset-cache-url http://localhost:8000/reset_prefix_cache \
  --reset-cache-between-rounds

# 执行流程:
# 1. 并发5 - 第1轮 (无缓存)
# 2. [重置缓存]
# 3. 并发5 - 第2轮 (无缓存)
# 4. 并发10 - 第1轮 (可能使用并发5的缓存)
# 5. [重置缓存]
# 6. 并发10 - 第2轮 (无缓存)
# 7. 并发20 - 第1轮 (可能使用并发10的缓存)
# 8. [重置缓存]
# 9. 并发20 - 第2轮 (无缓存)


# 示例 2: 仅在并发层级之间重置 KVCache
# 适用场景: 测试每个并发层级的缓存性能，确保不同并发层级间独立测试
python dual_round_benchmarker.py \
  --dataset data/test.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 20 \
  --reset-cache-url http://localhost:8000/reset_prefix_cache \
  --reset-cache-between-concurrency

# 执行流程:
# 1. 并发5 - 第1轮 (无缓存)
# 2. 并发5 - 第2轮 (使用第1轮缓存)
# 3. [重置缓存]
# 4. 并发10 - 第1轮 (无缓存)
# 5. 并发10 - 第2轮 (使用第1轮缓存)
# 6. [重置缓存]
# 7. 并发20 - 第1轮 (无缓存)
# 8. 并发20 - 第2轮 (使用第1轮缓存)


# 示例 3: 同时在轮次和并发层级之间重置 KVCache
# 适用场景: 确保每次测试都是完全干净的环境
python dual_round_benchmarker.py \
  --dataset data/test.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 20 \
  --reset-cache-url http://localhost:8000/reset_prefix_cache \
  --reset-cache-between-rounds \
  --reset-cache-between-concurrency

# 执行流程:
# 1. 并发5 - 第1轮 (无缓存)
# 2. [重置缓存]
# 3. 并发5 - 第2轮 (无缓存)
# 4. [重置缓存]
# 5. 并发10 - 第1轮 (无缓存)
# 6. [重置缓存]
# 7. 并发10 - 第2轮 (无缓存)
# 8. [重置缓存]
# 9. 并发20 - 第1轮 (无缓存)
# 10. [重置缓存]
# 11. 并发20 - 第2轮 (无缓存)


# 示例 4: 不重置 KVCache (默认行为)
# 适用场景: 测试完整的缓存累积效果
python dual_round_benchmarker.py \
  --dataset data/test.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 20

# 执行流程 (无重置):
# 1. 并发5 - 第1轮 (无缓存)
# 2. 并发5 - 第2轮 (使用第1轮缓存)
# 3. 并发10 - 第1轮 (使用并发5的缓存)
# 4. 并发10 - 第2轮 (使用所有之前的缓存)
# 5. 并发20 - 第1轮 (使用所有之前的缓存)
# 6. 并发20 - 第2轮 (使用所有之前的缓存)


# 示例 5: 完整功能测试（包含 Prometheus 和 SLO）
python dual_round_benchmarker.py \
  --dataset data/test.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 100 \
  --concurrency 5 10 20 50 \
  --max-output-tokens 256 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate memory_usage_bytes \
  --slo-file examples/slo_example.yaml \
  --reset-cache-url http://localhost:8000/reset_prefix_cache \
  --reset-cache-between-rounds \
  --save-requests \
  --output-dir results
