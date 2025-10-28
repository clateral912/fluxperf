#!/usr/bin/env python3
"""
Test vLLM Manager functionality
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxperf.vllm_manager import VLLMManager


def test_vllm_manager_initialization():
    """Test VLLMManager initialization"""
    print("Testing VLLMManager initialization...")
    
    manager = VLLMManager()
    assert manager.process is None
    
    print("✓ VLLMManager initialization test passed")


@patch('subprocess.Popen')
def test_start_vllm(mock_popen):
    """Test vLLM startup"""
    print("Testing vLLM startup...")
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.pid = 12345
    mock_popen.return_value = mock_process
    
    manager = VLLMManager()
    command = "vllm serve model --port 8000"
    
    manager.start_vllm(command, port=8000)
    
    assert manager.process is not None
    mock_popen.assert_called_once()
    
    print("✓ vLLM startup test passed")


@patch('requests.get')
def test_wait_for_health_success(mock_get):
    """Test health check - success case"""
    print("Testing health check (success)...")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    manager = VLLMManager()
    
    result = manager.wait_for_health("http://0.0.0.0", 8000, timeout=5)
    
    assert result is True
    
    print("✓ Health check success test passed")


@patch('requests.get')
@patch('time.sleep')
def test_wait_for_health_timeout(mock_sleep, mock_get):
    """Test health check - timeout case"""
    print("Testing health check (timeout)...")
    
    import requests
    mock_get.side_effect = requests.exceptions.RequestException("Connection refused")
    
    manager = VLLMManager()
    
    result = manager.wait_for_health("http://0.0.0.0", 8000, timeout=1)
    
    assert result is False
    
    print("✓ Health check timeout test passed")


@patch('subprocess.run')
def test_stop_vllm(mock_run):
    """Test vLLM shutdown"""
    print("Testing vLLM shutdown...")
    
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="500\n"
    )
    
    manager = VLLMManager()
    manager.process = MagicMock()
    
    manager.stop_vllm()
    
    assert manager.process is None
    
    print("✓ vLLM shutdown test passed")




@patch('subprocess.run')
@patch('subprocess.Popen')
@patch('requests.get')
def test_restart_vllm(mock_get, mock_popen, mock_run):
    """Test vLLM restart"""
    print("Testing vLLM restart...")
    
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.pid = 12345
    mock_popen.return_value = mock_process
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    mock_run.return_value = MagicMock(
        stdout="500\n",
        returncode=0
    )
    
    manager = VLLMManager()
    command = "vllm serve model --port 8000"
    
    result = manager.restart_vllm(command, port=8000, timeout=5)
    
    assert result is True
    
    print("✓ vLLM restart test passed")




if __name__ == '__main__':
    try:
        test_vllm_manager_initialization()
        test_start_vllm()
        test_wait_for_health_success()
        test_wait_for_health_timeout()
        test_stop_vllm()
        test_restart_vllm()
        
        print("\n" + "=" * 60)
        print("All VLLMManager tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
