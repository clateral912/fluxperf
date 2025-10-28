import asyncio
import time
from typing import Dict, List, Tuple

import aiohttp
from prometheus_client.parser import text_string_to_metric_families


class PrometheusCollector:
    def __init__(self, prometheus_url: str, metric_names: List[str]):
        self.prometheus_url = prometheus_url
        self.metric_names = metric_names
        self.collected_data: Dict[str, List[Tuple[float, float]]] = {name: [] for name in metric_names}
    
    async def fetch_metrics(self, session: aiohttp.ClientSession) -> Dict[str, List[float]]:
        try:
            async with session.get(self.prometheus_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print(f"Warning: Prometheus endpoint returned status code {response.status}")
                    return {}
                
                text = await response.text()
                current_time = time.time()
                metrics_data = {}
                
                for family in text_string_to_metric_families(text):
                    if family.name in self.metric_names:
                        values = []
                        for sample in family.samples:
                            if sample.name == family.name or sample.name.startswith(f"{family.name}_"):
                                timestamp = sample.timestamp if sample.timestamp else current_time
                                values.append((timestamp, sample.value))
                        
                        if values:
                            metrics_data[family.name] = values
                
                return metrics_data
        except Exception as e:
            print(f"Warning: Failed to fetch metrics from Prometheus: {e}")
            return {}
    
    async def collect_during_test(self, session: aiohttp.ClientSession, start_time: float, end_time: float, interval: float = 1.0):
        while time.time() < end_time:
            metrics_data = await self.fetch_metrics(session)
            current_time = time.time()
            
            for metric_name, values in metrics_data.items():
                for timestamp, value in values:
                    self.collected_data[metric_name].append((timestamp, value))
            
            await asyncio.sleep(interval)
    
    def get_metrics_in_timerange(self, start_time: float, end_time: float) -> Dict[str, List[float]]:
        filtered_metrics = {}
        
        for metric_name in self.metric_names:
            values = []
            for timestamp, value in self.collected_data[metric_name]:
                if start_time <= timestamp <= end_time:
                    values.append(value)
            
            if values:
                filtered_metrics[metric_name] = values
            else:
                print(f"Warning: No data found for metric '{metric_name}' within test time range")
        
        return filtered_metrics
    
    def clear_data(self):
        for metric_name in self.metric_names:
            self.collected_data[metric_name] = []
