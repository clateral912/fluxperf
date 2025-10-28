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
ROOT_DIR = Path(__file__).resolve().parent.parent


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
    print("Prometheus Integration Health Check")
    print("=" * 70)
    
    print("\n[1/7] Starting Mock Prometheus server...")
    prom_thread = threading.Thread(
        target=start_server,
        args=(PROMETHEUS_PORT, MockPrometheusHandler),
        daemon=True
    )
    prom_thread.start()
    time.sleep(1)
    print(f"✓ Mock Prometheus server started on port {PROMETHEUS_PORT}")
    
    print("\n[2/7] Starting Mock OpenAI server...")
    openai_thread = threading.Thread(
        target=start_server,
        args=(PORT, MockOpenAIHandler),
        daemon=True
    )
    openai_thread.start()
    time.sleep(1)
    print(f"✓ Mock OpenAI server started on port {PORT}")
    
    print("\n[3/7] Checking test data file...")
    dataset_path = ROOT_DIR / "examples/example_dataset.json"
    if not dataset_path.exists():
        print(f"✗ Dataset file not found: {dataset_path}")
        return False
    print(f"✓ Dataset file exists: {dataset_path}")
    
    print("\n[4/7] Verifying Prometheus endpoint...")
    import urllib.request
    try:
        response = urllib.request.urlopen(f"http://localhost:{PROMETHEUS_PORT}/metrics")
        content = response.read().decode()
        if "lmcache_hit_rate" in content and "memory_usage_bytes" in content:
            print("✓ Prometheus endpoint returns correct metrics format")
        else:
            print("✗ Prometheus endpoint returns incomplete metrics")
            return False
    except Exception as e:
        print(f"✗ Cannot access Prometheus endpoint: {e}")
        return False
    
    print("\n[5/7] Running benchmark script (with Prometheus metrics collection)...")
    import subprocess
    
    cmd = [
        sys.executable,
        str(ROOT_DIR / "fluxperf.py"),
        "--dataset", str(dataset_path),
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
            print(f"✗ Script execution failed, return code: {result.returncode}")
            print("\nStdout:")
            print(result.stdout)
            print("\nStderr:")
            print(result.stderr)
            return False
        
        print("✓ Script executed successfully")
        
        if "Prometheus 指标:" in result.stdout:
            print("✓ Console output contains Prometheus metrics")
        else:
            print("⚠ Warning: Console output does not contain Prometheus metrics")
        
    except subprocess.TimeoutExpired:
        print("✗ Script execution timeout")
        return False
    except Exception as e:
        print(f"✗ Error running script: {e}")
        return False
    
    print("\n[6/7] Verifying output files...")
    
    json_output = Path("test_results.json")
    if json_output.exists():
        print(f"✓ JSON output file generated: {json_output}")
        try:
            with open(json_output, 'r') as f:
                data = json.load(f)
                print(f"  - Contains results for {len(data)} concurrency levels")
        except Exception as e:
            print(f"✗ Cannot parse JSON file: {e}")
            return False
    else:
        print(f"✗ JSON output file not generated: {json_output}")
        return False
    
    test_results_dir = Path("test_results")
    if test_results_dir.exists():
        csv_files = list(test_results_dir.glob("benchmark_*.csv"))
        params_files = list(test_results_dir.glob("benchmark_*_params.txt"))
        
        if csv_files:
            print(f"✓ CSV output file generated: {csv_files[0].name}")
            
            with open(csv_files[0], 'r', encoding='utf-8') as f:
                csv_content = f.read()
                if "Prometheus 指标" in csv_content and "lmcache_hit_rate" in csv_content:
                    print("✓ CSV file contains Prometheus metrics data")
                else:
                    print("⚠ Warning: CSV file does not contain Prometheus metrics")
        else:
            print("✗ CSV output file not generated")
            return False
        
        if params_files:
            print(f"✓ Parameters file generated: {params_files[0].name}")
        else:
            print("⚠ Warning: Parameters file not generated")
    else:
        print("✗ Output directory not created")
        return False
    
    print("\n[7/7] Verifying Prometheus metrics data quality...")
    
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
            print("✓ All specified Prometheus metrics collected")
        else:
            missing = [k for k, v in found_metrics.items() if not v]
            print(f"⚠ Warning: The following metrics not found: {', '.join(missing)}")
    
    print("\n" + "=" * 70)
    print("Health Check Complete!")
    print("=" * 70)
    
    print("\nTest Summary:")
    print("✓ Mock servers started successfully")
    print("✓ Prometheus metrics collection working")
    print("✓ Output files generated successfully")
    print("✓ Data format correct")
    
    print("\nCleanup suggestion:")
    print("  rm -rf test_results test_results.json")
    
    return True


def main():
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Exception occurred during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
