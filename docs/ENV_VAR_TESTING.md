# Environment Variable Testing Guide

## Overview

The Recipe feature supports setting independent environment variables for each stage, automatically restoring original values after stage completion.

## Testing Methods

### 1. Unit Testing

Run unit tests for environment variable logic:

```bash
python tests/test_env_variables.py
```

**Test Coverage**:
- ✓ Environment variables set correctly before stage starts
- ✓ Environment variables restored correctly after stage ends
- ✓ Environment variables from multiple stages don't interfere
- ✓ Nested environment variable scenarios

### 2. Integration Testing

Test environment variable management using real recipe files:

```bash
# Method 1: Use test script
bash tests/test_recipe_env_integration.sh

# Method 2: Manually run recipe
python fluxperf.py --recipe recipe_env_test.yaml
```

While running, you can monitor environment variables in another terminal:

```bash
# After starting recipe, run in another terminal
watch -n 1 'ps aux | grep fluxperf | head -1'
```

### 3. Viewing Stage Names

Now CLI output table headers show stage names instead of concurrency:

**Output When Using Recipe**:
```
            Test Stage 1 - GPU 0 - Round 1
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳...
```

**Output When Not Using Recipe**:
```
   FluxPerf | LLM Metrics (Concurrency: 2, Round: 1)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳...
```

## Test Recipe Example

`recipe_env_test.yaml` contains 3 stages, each setting different environment variables:

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

## Verifying Environment Variables Take Effect

### Method 1: Check in Code

Modify your application code to print environment variables:

```python
import os
print(f"Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
```

### Method 2: Using Wrapper Script

Create a wrapper script to capture environment variables:

```bash
#!/bin/bash
# save_env.sh
echo "Stage environment variables:" >> /tmp/stage_env.log
env | grep -E "CUDA|TEST_|VLLM" >> /tmp/stage_env.log
echo "---" >> /tmp/stage_env.log

# Call actual program
exec "$@"
```

Then use this wrapper in your service.

### Method 3: View Process Environment Variables

```bash
# Find process PID
ps aux | grep your_service

# View environment variables
cat /proc/<PID>/environ | tr '\0' '\n'
```

## Common Use Cases

### 1. Testing Different GPU Configurations

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

### 2. Testing Different Backends

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

### 3. Testing Cache Toggle

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

## Troubleshooting

### Environment Variables Not Taking Effect

1. **Check if application reads environment variables**: Confirm your app actually reads configuration from environment variables
2. **Check if restart needed**: Some applications only read environment variables at startup
3. **Check shell expansion**: Environment variable values are converted to strings, special characters need escaping

### Environment Variables Not Restored

1. **Check exception handling**: If stage fails midway, finally block still restores environment variables
2. **Check subprocesses**: Inherited environment variables in subprocesses don't affect parent process

### Stage Names Not Displayed

Ensure you're using the `--recipe` parameter rather than command-line parameters. Stage names only display when running through recipe.

## Best Practices

1. **Use Descriptive Names**: Stage names should clearly describe test purpose
2. **Minimize Variable Count**: Only set environment variables that need changing
3. **Document Side Effects**: Explain environment variable impact in recipe comments
4. **Test Recovery**: Add verification steps in test recipe to ensure variables restore correctly

## Debugging Tips

### Print All Environment Variables

Add to the stage loop in `run_recipe_benchmark` function:

```python
print("\nCurrent environment variables:")
for key in sorted(os.environ.keys()):
    if any(k in key for k in ['CUDA', 'VLLM', 'TEST']):
        print(f"  {key} = {os.environ[key]}")
```

### Save Environment Variable Snapshot

```python
import json

# Before stage
before = dict(os.environ)

# After stage
after = dict(os.environ)

# Compare
diff = {k: (before.get(k), after.get(k)) 
        for k in set(before) | set(after) 
        if before.get(k) != after.get(k)}

print(json.dumps(diff, indent=2))
```
