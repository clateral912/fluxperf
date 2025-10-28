import subprocess
import time
from typing import Optional

import requests


class VLLMManager:
    """Manages vLLM server lifecycle with auto-restart functionality."""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
    
    def start_vllm(self, command: str, port: int) -> None:
        """
        Start vLLM server using the provided bash command.
        
        Args:
            command: Bash command to start vLLM server
            port: Port number where vLLM will listen
        """
        print(f"Starting vLLM server on port {port}...")
        print(f"Command: {command}")
        
        self.process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"vLLM process started with PID: {self.process.pid}")
    
    def wait_for_health(self, endpoint: str, port: int, timeout: int = 300) -> bool:
        """
        Poll the vLLM health endpoint until it returns 200 or timeout is reached.
        
        Args:
            endpoint: Base endpoint URL (e.g., "http://0.0.0.0")
            port: Port number
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if health check succeeded, False if timeout
        """
        health_url = f"{endpoint}:{port}/health"
        print(f"Waiting for vLLM health check at {health_url}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print(f"vLLM server is healthy and ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        print(f"Warning: vLLM health check timed out after {timeout} seconds")
        return False
    
    def stop_vllm(self) -> None:
        """
        Stop vLLM server and wait for GPU memory to be released.
        Uses pkill to terminate all vLLM processes and monitors GPU memory.
        """
        print("Stopping vLLM server...")
        
        # Kill vLLM processes
        try:
            subprocess.run(
                ["pkill", "-f", "[V]LLM"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Sent kill signal to vLLM processes")
        except Exception as e:
            print(f"Warning: Error while killing vLLM processes: {e}")
        
        # Wait for process to terminate
        if self.process:
            try:
                self.process.wait(timeout=10)
                print("vLLM process terminated")
            except subprocess.TimeoutExpired:
                print("Warning: vLLM process did not terminate cleanly, forcing kill")
                self.process.kill()
            self.process = None
        
        # Monitor GPU memory until it drops below 1 GiB
        print("Waiting for GPU memory to be released (< 1 GiB)...")
        max_wait = 60  # Maximum 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                gpu_memory_mb = [float(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
                max_memory_mb = max(gpu_memory_mb) if gpu_memory_mb else 0
                max_memory_gb = max_memory_mb / 1024
                
                print(f"GPU memory: {max_memory_gb:.2f} GiB")
                
                if max_memory_gb < 1.0:
                    print("GPU memory released successfully")
                    return
                
            except Exception as e:
                print(f"Warning: Error checking GPU memory: {e}")
            
            time.sleep(2)
        
        print(f"Warning: GPU memory did not drop below 1 GiB within {max_wait} seconds")
    
    def restart_vllm(self, command: str, port: int, endpoint: str = "http://0.0.0.0", timeout: int = 300) -> bool:
        """
        Restart vLLM server by stopping and then starting it.
        
        Args:
            command: Bash command to start vLLM server
            port: Port number
            endpoint: Base endpoint URL for health check
            timeout: Maximum time to wait for health check
            
        Returns:
            True if restart succeeded and health check passed, False otherwise
        """
        print("\n" + "="*80)
        print("RESTARTING vLLM SERVER")
        print("="*80 + "\n")
        
        self.stop_vllm()
        time.sleep(5)  # Additional grace period
        self.start_vllm(command, port)
        
        return self.wait_for_health(endpoint, port, timeout)
