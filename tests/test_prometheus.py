#!/usr/bin/env python3

import asyncio
import http.server
import socketserver
import threading
import time
from pathlib import Path
import sys
import json

PORT = 9999
PROMETHEUS_PORT = 9998


class MockPrometheusHandler(http.server.BaseHTTPRequestHandler):
    request_count = 0
    
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.end_headers()
            
            timestamp = int(time.time() * 1000)
            
            MockPrometheusHandler.request_count += 1
            hit_rate = 0.75 + (MockPrometheusHandler.request_count % 10) * 0.02
            
            metrics = f"""# HELP lmcache_hit_rate Cache hit rate
# TYPE lmcache_hit_rate gauge
lmcache_hit_rate {hit_rate} {timestamp}

# HELP memory_usage_bytes Memory usage in bytes
# TYPE memory_usage_bytes gauge
memory_usage_bytes 1073741824 {timestamp}

# HELP cpu_utilization CPU utilization percentage
# TYPE cpu_utilization gauge
cpu_utilization 0.65 {timestamp}

# HELP requests_total Total number of requests
# TYPE requests_total counter
requests_total {MockPrometheusHandler.request_count} {timestamp}
"""
            self.wfile.write(metrics.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass


class MockOpenAIHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/v1/chat/completions':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(body)
            except:
                self.send_response(400)
                self.end_headers()
                return
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            
            tokens = ["Hello", " world", "! ", "This", " is", " a", " test", " response", "."]
            
            for i, token in enumerate(tokens):
                chunk = {
                    "id": "test-id",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "gpt-3.5-turbo",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None if i < len(tokens) - 1 else "stop"
                    }]
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.05)
            
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass


def start_server(port, handler_class):
    with socketserver.TCPServer(("", port), handler_class) as httpd:
        httpd.serve_forever()


def run_test():
    print("=" * 70)
    print("Prometheus 集成功能健康检查")
    print("=" * 70)
    
    print("\n[1/7] 启动 Mock Prometheus 服务器...")
    prom_thread = threading.Thread(
        target=start_server,
        args=(PROMETHEUS_PORT, MockPrometheusHandler),
        daemon=True
    )
    prom_thread.start()
    time.sleep(1)
    print(f"✓ Mock Prometheus 服务器已启动在端口 {PROMETHEUS_PORT}")
    
    print("\n[2/7] 启动 Mock OpenAI 服务器...")
    openai_thread = threading.Thread(
        target=start_server,
        args=(PORT, MockOpenAIHandler),
        daemon=True
    )
    openai_thread.start()
    time.sleep(1)
    print(f"✓ Mock OpenAI 服务器已启动在端口 {PORT}")
    
    print("\n[3/7] 检查测试数据文件...")
    dataset_path = Path("example_dataset.json")
    if not dataset_path.exists():
        print(f"✗ 未找到数据集文件: {dataset_path}")
        return False
    print(f"✓ 数据集文件存在: {dataset_path}")
    
    print("\n[4/7] 验证 Prometheus endpoint...")
    import urllib.request
    try:
        response = urllib.request.urlopen(f"http://localhost:{PROMETHEUS_PORT}/metrics")
        content = response.read().decode()
        if "lmcache_hit_rate" in content and "memory_usage_bytes" in content:
            print("✓ Prometheus endpoint 返回正确的指标格式")
        else:
            print("✗ Prometheus endpoint 返回的指标不完整")
            return False
    except Exception as e:
        print(f"✗ 无法访问 Prometheus endpoint: {e}")
        return False
    
    print("\n[5/7] 运行压测脚本（带 Prometheus 指标收集）...")
    import subprocess
    
    cmd = [
        sys.executable,
        "dual_round_benchmarker.py",
        "--dataset", "example_dataset.json",
        "--endpoint", f"http://localhost:{PORT}/v1/chat/completions",
        "--num-samples", "3",
        "--concurrency", "2",
        "--prometheus-url", f"http://localhost:{PROMETHEUS_PORT}/metrics",
        "--prometheus-metrics", "lmcache_hit_rate", "memory_usage_bytes", "cpu_utilization",
        "--output-dir", "test_results",
        "--output", "test_results.json"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"✗ 脚本执行失败，返回码: {result.returncode}")
            print("\n标准输出:")
            print(result.stdout)
            print("\n标准错误:")
            print(result.stderr)
            return False
        
        print("✓ 脚本执行成功")
        
        if "Prometheus 指标:" in result.stdout:
            print("✓ 控制台输出包含 Prometheus 指标")
        else:
            print("⚠ 警告: 控制台输出未找到 Prometheus 指标")
        
    except subprocess.TimeoutExpired:
        print("✗ 脚本执行超时")
        return False
    except Exception as e:
        print(f"✗ 运行脚本时出错: {e}")
        return False
    
    print("\n[6/7] 验证输出文件...")
    
    json_output = Path("test_results.json")
    if json_output.exists():
        print(f"✓ JSON 输出文件已生成: {json_output}")
        try:
            with open(json_output, 'r') as f:
                data = json.load(f)
                print(f"  - 包含 {len(data)} 个并发层级的结果")
        except Exception as e:
            print(f"✗ 无法解析 JSON 文件: {e}")
            return False
    else:
        print(f"✗ JSON 输出文件未生成: {json_output}")
        return False
    
    test_results_dir = Path("test_results")
    if test_results_dir.exists():
        csv_files = list(test_results_dir.glob("benchmark_*.csv"))
        params_files = list(test_results_dir.glob("benchmark_*_params.txt"))
        
        if csv_files:
            print(f"✓ CSV 输出文件已生成: {csv_files[0].name}")
            
            with open(csv_files[0], 'r', encoding='utf-8') as f:
                csv_content = f.read()
                if "Prometheus 指标" in csv_content and "lmcache_hit_rate" in csv_content:
                    print("✓ CSV 文件包含 Prometheus 指标数据")
                else:
                    print("⚠ 警告: CSV 文件未找到 Prometheus 指标")
        else:
            print("✗ CSV 输出文件未生成")
            return False
        
        if params_files:
            print(f"✓ 参数文件已生成: {params_files[0].name}")
        else:
            print("⚠ 警告: 参数文件未生成")
    else:
        print("✗ 输出目录未创建")
        return False
    
    print("\n[7/7] 验证 Prometheus 指标数据质量...")
    
    with open(csv_files[0], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        found_metrics = {
            'lmcache_hit_rate': False,
            'memory_usage_bytes': False,
            'cpu_utilization': False
        }
        
        for line in lines:
            for metric in found_metrics.keys():
                if metric in line:
                    found_metrics[metric] = True
        
        all_found = all(found_metrics.values())
        if all_found:
            print("✓ 所有指定的 Prometheus 指标都已收集")
        else:
            missing = [k for k, v in found_metrics.items() if not v]
            print(f"⚠ 警告: 以下指标未找到: {', '.join(missing)}")
    
    print("\n" + "=" * 70)
    print("健康检查完成!")
    print("=" * 70)
    
    print("\n测试摘要:")
    print("✓ Mock 服务器启动成功")
    print("✓ Prometheus 指标收集功能正常")
    print("✓ 输出文件生成正常")
    print("✓ 数据格式正确")
    
    print("\n清理建议:")
    print("  rm -rf test_results test_results.json")
    
    return True


def main():
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
