import json
import random
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import yaml

from .models import (
    BenchmarkConfig,
    BenchmarkMode,
    Recipe,
    RecipeStage,
    RecipeSuite,
    RequestMetrics,
    SLOConstraints,
)


class DatasetLoader:
    @staticmethod
    def load_dataset(path: Path) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
                else:
                    raise ValueError("Unsupported JSON format")
            elif path.suffix == '.jsonl':
                return [json.loads(line) for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    def sample_entries(dataset: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
        if not dataset or num_samples <= 0:
            return []
        if num_samples >= len(dataset):
            return dataset
        return random.sample(dataset, num_samples)

    @staticmethod
    def truncate_text(text: str, max_length: Optional[int]) -> str:
        if max_length is None:
            return text
        if len(text) > max_length:
            return text[:max_length]
        return text

    @staticmethod
    def is_multi_turn_entry(entry: Dict[str, Any]) -> bool:
        if isinstance(entry.get("user_messages"), list):
            return True
        messages = entry.get("messages")
        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            return True
        return False


class SLOLoader:
    @staticmethod
    def load_slo(slo_file: Path) -> Optional[SLOConstraints]:
        if slo_file is None or not slo_file.exists():
            return None
        
        with open(slo_file, 'r', encoding='utf-8') as f:
            slo_data = yaml.safe_load(f)
        
        if not slo_data or 'constraints' not in slo_data:
            return None
        
        constraints_data = slo_data['constraints']
        
        return SLOConstraints(
            ttft_ms=constraints_data.get('ttft_ms', {}).get('max'),
            itl_ms=constraints_data.get('itl_ms', {}).get('max'),
            latency_ms=constraints_data.get('latency_ms', {}).get('max'),
            output_token_throughput=constraints_data.get('output_token_throughput', {}).get('min')
        )
    
    @staticmethod
    def check_slo(metrics: RequestMetrics, slo: Optional[SLOConstraints]) -> bool:
        if slo is None:
            return True
        
        if metrics.error is not None:
            return False
        
        ttft_ms = metrics.time_to_first_token * 1000
        if slo.ttft_ms is not None and ttft_ms > slo.ttft_ms:
            return False
        
        if metrics.inter_token_latencies:
            avg_itl_ms = mean(metrics.inter_token_latencies)
            if slo.itl_ms is not None and avg_itl_ms > slo.itl_ms:
                return False
        
        latency_ms = metrics.total_latency * 1000
        if slo.latency_ms is not None and latency_ms > slo.latency_ms:
            return False
        
        if slo.output_token_throughput is not None and metrics.throughput < slo.output_token_throughput:
            return False
        
        return True
    
    @staticmethod
    def validate_slo(metrics: RequestMetrics, slo: Optional[SLOConstraints]) -> bool:
        return SLOLoader.check_slo(metrics, slo)


class RecipeLoader:
    @staticmethod
    def load_recipe(recipe_file: Path) -> Recipe:
        if not recipe_file.exists():
            raise ValueError(f"Recipe file does not exist: {recipe_file}")
        
        with open(recipe_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError("Recipe file format error: root element must be a dictionary")
        
        global_config = data.get('global', {})
        if not isinstance(global_config, dict):
            raise ValueError("Recipe file format error: 'global' must be a dictionary")
        
        suites_data = data.get('suites')
        raw_stages = data.get('stages') if suites_data is None else None

        suites: List[RecipeSuite] = []
        stages: List[RecipeStage] = []

        if suites_data is not None:
            if not isinstance(suites_data, list):
                raise ValueError("Recipe file format error: 'suites' must be a list")
            for suite_idx, suite_data in enumerate(suites_data):
                if not isinstance(suite_data, dict):
                    raise ValueError(f"Recipe file format error: suite {suite_idx} must be a dictionary")
                suite_name = suite_data.get('name', f'Suite {suite_idx + 1}')
                stages_data = suite_data.get('stages', [])
                if not isinstance(stages_data, list):
                    raise ValueError(f"Recipe file format error: 'stages' in suite '{suite_name}' must be a list")

                suite_stages: List[RecipeStage] = []
                for stage_idx, stage_data in enumerate(stages_data):
                    stage = RecipeLoader._parse_stage(stage_data, stage_idx, suite_name)
                    suite_stages.append(stage)
                    stages.append(stage)

                suites.append(RecipeSuite(name=suite_name, stages=suite_stages))
        else:
            stages_data = raw_stages or []
            if not isinstance(stages_data, list):
                raise ValueError("Recipe file format error: 'stages' must be a list")
            default_suite = RecipeSuite(name="Default Suite")
            for stage_idx, stage_data in enumerate(stages_data):
                stage = RecipeLoader._parse_stage(stage_data, stage_idx, default_suite.name)
                default_suite.stages.append(stage)
                stages.append(stage)
            suites.append(default_suite)

        mock_server = data.get('mock_server')

        return Recipe(
            global_config=global_config,
            stages=stages,
            mock_server=mock_server,
            suites=suites if suites else None
        )

    @staticmethod
    def _parse_stage(stage_data: Any, index: int, suite_name: Optional[str]) -> RecipeStage:
        if not isinstance(stage_data, dict):
            raise ValueError(f"Recipe file format error: stage {index} must be a dictionary")

        name = stage_data.get('name', f'Stage {index + 1}')
        concurrency_levels = stage_data.get('concurrency_levels', [])
        num_samples = stage_data.get('num_samples', [])
        env = stage_data.get('env', {})
        dataset = stage_data.get('dataset')
        max_output_tokens = stage_data.get('max_output_tokens')
        min_output_tokens = stage_data.get('min_output_tokens')

        if not isinstance(concurrency_levels, list):
            raise ValueError(f"Recipe file format error: concurrency_levels in stage '{name}' must be a list")
        if not isinstance(num_samples, list):
            raise ValueError(f"Recipe file format error: num_samples in stage '{name}' must be a list")
        if not isinstance(env, dict):
            raise ValueError(f"Recipe file format error: env in stage '{name}' must be a dictionary")

        stage = RecipeStage(
            name=name,
            concurrency_levels=concurrency_levels,
            num_samples=num_samples,
            dataset=dataset,
            max_output_tokens=max_output_tokens,
            min_output_tokens=min_output_tokens,
            env=env,
            suite_name=suite_name
        )
        return stage
    
    @staticmethod
    def create_config_from_recipe(recipe: Recipe, stage: RecipeStage) -> BenchmarkConfig:
        g = recipe.global_config
        
        mode_str = g.get('mode', 'multi_turn')
        try:
            mode = BenchmarkMode(mode_str)
        except ValueError:
            raise ValueError(f"Unsupported mode: {mode_str}. Supported modes: dual_round, multi_turn")
        
        dataset_value = stage.dataset or g.get('dataset')
        if not dataset_value:
            raise ValueError("Missing dataset configuration in recipe")

        dataset_path = Path(dataset_value)
        
        endpoint = g.get('endpoint')
        if not endpoint and not (recipe.mock_server and recipe.mock_server.get('enabled')):
            raise ValueError("Missing endpoint configuration in recipe and mock_server is not enabled")
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        max_output_tokens = stage.max_output_tokens if stage.max_output_tokens is not None else g.get('max_output_tokens')
        min_output_tokens = stage.min_output_tokens if stage.min_output_tokens is not None else g.get('min_output_tokens')

        return BenchmarkConfig(
            dataset_path=dataset_path,
            endpoint_url=endpoint or "",
            num_samples=stage.num_samples,
            concurrency_levels=stage.concurrency_levels,
            mode=mode,
            max_input_length=g.get('max_input_length'),
            max_output_tokens=max_output_tokens,
            min_output_tokens=min_output_tokens,
            model_name=g.get('model', 'gpt-3.5-turbo'),
            api_key=g.get('api_key'),
            timeout=g.get('timeout', 300),
            shuffle_round2=g.get('shuffle_round2', True),
            slo_file=Path(g['slo_file']) if g.get('slo_file') else None,
            output_dir=Path(g.get('output_dir', 'benchmark_results')),
            prometheus_url=g.get('prometheus_url'),
            prometheus_metrics=g.get('prometheus_metrics', []),
            save_requests=g.get('save_requests', False),
            reset_cache_url=g.get('reset_cache_url'),
            reset_cache_between_rounds=g.get('reset_cache_between_rounds', False),
            reset_cache_between_concurrency=g.get('reset_cache_between_concurrency', False),
            debug=g.get('debug', False),
            debug_verbose=g.get('debug_verbose', False),
            debug_log_dir=Path(g['debug_log_dir']) if g.get('debug_log_dir') else None,
            max_context_tokens=g.get('max_context_tokens'),
            tokenizer_name=g.get('tokenizer_name'),
            tokenizer_trust_remote_code=g.get('tokenizer_trust_remote_code', False),
            tokenizer_revision=g.get('tokenizer_revision'),
            suite_name=stage.suite_name,
            stage_name=stage.name,
            run_timestamp=timestamp
        )
