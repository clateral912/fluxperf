import argparse
import asyncio
from pathlib import Path

from .analyzer import MetricsAnalyzer
from .loaders import RecipeLoader, SLOLoader
from .models import BenchmarkConfig, BenchmarkMode
from .runner import BenchmarkRunner, run_recipe_benchmark


def add_cli_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--mock-server",
        action="store_true",
        help="Start built-in Mock LLM server"
    )
    parser.add_argument(
        "--mock-host",
        type=str,
        default="0.0.0.0",
        help="Mock server listen address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--mock-port",
        type=int,
        default=8001,
        help="Mock server port (default: 8001)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dual-round benchmark tool - Send two rounds of identical requests to OpenAI-format API and collect performance metrics"
    )
    
    add_cli_arguments(parser)

    parser.add_argument(
        "--recipe",
        type=Path,
        default=None,
        help="Recipe configuration file path (YAML format). When using this parameter, other parameters will be ignored"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=['dual_round', 'multi_turn'],
        default='multi_turn',
        help="Test mode: dual_round (two rounds of single requests) or multi_turn (multi-turn conversation)"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=False,
        help="Dataset file path (JSON or JSONL format)"
    )
    
    parser.add_argument(
        "--endpoint",
        type=str,
        required=False,
        default=None,
        help="OpenAI-format API endpoint URL (optional when using --mock-server)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        nargs='+',
        required=False,
        help="Number of samples randomly drawn from dataset. Multiple values can be specified for different concurrency levels. When count doesn't match --concurrency, defaults to 2*concurrency for each level"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs='+',
        default=[10],
        help="Request concurrency, multiple values can be specified (default: 10), e.g.: --concurrency 5 10 50 100"
    )
    
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help="Maximum input length, inputs exceeding this will be truncated"
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum output token count, controls generation length (OSL)"
    )

    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=None,
        help="Minimum output token count"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model name (default: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (if required)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--no-shuffle-round2",
        action="store_true",
        help="Don't shuffle sample order in round 2"
    )
    
    parser.add_argument(
        "--slo-file",
        type=Path,
        default=None,
        help="SLO configuration file path (YAML format)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Results output directory (default: benchmark_results)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="JSON results output file path (default: benchmark_results.json)"
    )
    
    parser.add_argument(
        "--prometheus-url",
        type=str,
        default=None,
        help="Prometheus metrics endpoint URL (e.g.: http://localhost:3001/metrics)"
    )
    
    parser.add_argument(
        "--prometheus-metrics",
        type=str,
        nargs='+',
        default=[],
        help="List of Prometheus metric names to collect (e.g.: lmcache_hit_rate memory_usage)"
    )

    parser.add_argument(
        "--save-requests",
        action="store_true",
        help="Save request data to local JSONL file for debugging"
    )

    parser.add_argument(
        "--reset-cache-url",
        type=str,
        default=None,
        help="vLLM KVCache reset endpoint URL (e.g.: http://localhost:8000/reset_prefix_cache)"
    )

    parser.add_argument(
        "--reset-cache-between-rounds",
        action="store_true",
        help="Reset KVCache between two rounds of tests for each concurrency level"
    )

    parser.add_argument(
        "--reset-cache-between-concurrency",
        action="store_true",
        help="Reset KVCache between different concurrency levels"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, save lightweight request lifecycle logs"
    )

    parser.add_argument(
        "--debug-verbose",
        action="store_true",
        help="Enable verbose debug mode, save complete request payloads (large files)"
    )

    parser.add_argument(
        "--debug-log-dir",
        type=Path,
        default=None,
        help="Debug log directory (default: debug_logs)"
    )

    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Maximum context token count, automatically truncates conversation history when exceeded"
    )

    args = parser.parse_args()
    
    # Recipe mode
    if args.recipe:
        try:
            recipe = RecipeLoader.load_recipe(args.recipe)
            print(f"Loaded recipe: {args.recipe}")
            print(f"Contains {len(recipe.stages)} test stages\n")

            # Allow CLI arguments to override any settings in recipe
            # Priority: command line arguments > recipe configuration
            overrides = []

            if args.dataset:
                recipe.global_config['dataset'] = str(args.dataset)
                overrides.append(f"dataset = {args.dataset}")

            if args.endpoint:
                recipe.global_config['endpoint'] = args.endpoint
                overrides.append(f"endpoint = {args.endpoint}")

            if args.mode != 'multi_turn':  # If not default value
                recipe.global_config['mode'] = args.mode
                overrides.append(f"mode = {args.mode}")

            if args.max_input_length is not None:
                recipe.global_config['max_input_length'] = args.max_input_length
                overrides.append(f"max_input_length = {args.max_input_length}")

            if args.max_output_tokens is not None:
                recipe.global_config['max_output_tokens'] = args.max_output_tokens
                overrides.append(f"max_output_tokens = {args.max_output_tokens}")

            if args.min_output_tokens is not None:
                recipe.global_config['min_output_tokens'] = args.min_output_tokens
                overrides.append(f"min_output_tokens = {args.min_output_tokens}")

            if args.model != "gpt-3.5-turbo":  # If not default value
                recipe.global_config['model'] = args.model
                overrides.append(f"model = {args.model}")

            if args.api_key is not None:
                recipe.global_config['api_key'] = args.api_key
                overrides.append(f"api_key = {args.api_key}")

            if args.timeout != 300:  # If not default value
                recipe.global_config['timeout'] = args.timeout
                overrides.append(f"timeout = {args.timeout}")

            if args.no_shuffle_round2:
                recipe.global_config['shuffle_round2'] = False
                overrides.append(f"shuffle_round2 = False")

            if args.slo_file is not None:
                recipe.global_config['slo_file'] = str(args.slo_file)
                overrides.append(f"slo_file = {args.slo_file}")

            if args.output_dir != Path("benchmark_results"):  # If not default value
                recipe.global_config['output_dir'] = str(args.output_dir)
                overrides.append(f"output_dir = {args.output_dir}")

            if args.prometheus_url is not None:
                recipe.global_config['prometheus_url'] = args.prometheus_url
                overrides.append(f"prometheus_url = {args.prometheus_url}")

            if args.prometheus_metrics:  # If not empty list
                recipe.global_config['prometheus_metrics'] = args.prometheus_metrics
                overrides.append(f"prometheus_metrics = {args.prometheus_metrics}")

            if args.save_requests:
                recipe.global_config['save_requests'] = True
                overrides.append(f"save_requests = True")

            if args.reset_cache_url is not None:
                recipe.global_config['reset_cache_url'] = args.reset_cache_url
                overrides.append(f"reset_cache_url = {args.reset_cache_url}")

            if args.reset_cache_between_rounds:
                recipe.global_config['reset_cache_between_rounds'] = True
                overrides.append(f"reset_cache_between_rounds = True")

            if args.reset_cache_between_concurrency:
                recipe.global_config['reset_cache_between_concurrency'] = True
                overrides.append(f"reset_cache_between_concurrency = True")

            if args.debug:
                recipe.global_config['debug'] = True
                overrides.append(f"debug = True")

            if args.debug_verbose:
                recipe.global_config['debug_verbose'] = True
                overrides.append(f"debug_verbose = True")

            if args.debug_log_dir is not None:
                recipe.global_config['debug_log_dir'] = str(args.debug_log_dir)
                overrides.append(f"debug_log_dir = {args.debug_log_dir}")

            if args.max_context_tokens is not None:
                recipe.global_config['max_context_tokens'] = args.max_context_tokens
                overrides.append(f"max_context_tokens = {args.max_context_tokens}")

            if overrides:
                print("CLI parameter overrides:")
                for override in overrides:
                    print(f"  - {override}")
                print()

            asyncio.run(run_recipe_benchmark(recipe))
            return
        except Exception as e:
            parser.error(f"Failed to load recipe: {e}")
    
    # Command line mode validation
    if not args.dataset:
        parser.error("Must provide --dataset parameter")
    if not args.num_samples:
        parser.error("Must provide --num-samples parameter")
    if not args.endpoint and not args.mock_server:
        parser.error("Must provide --endpoint parameter or enable --mock-server")
    
    if args.mock_server and not args.endpoint:
        args.endpoint = f"http://{args.mock_host}:{args.mock_port}/v1/chat/completions"

    num_samples_list = args.num_samples
    concurrency_levels = args.concurrency

    if len(num_samples_list) == len(concurrency_levels):
        final_num_samples = num_samples_list
        print(f"Using custom num_samples: {final_num_samples}")
    else:
        final_num_samples = [conc * 2 for conc in concurrency_levels]
        print(f"num_samples count doesn't match concurrency, using default rule (2*concurrency): {final_num_samples}")

    mode = BenchmarkMode(args.mode)
    
    slo = None
    if args.slo_file:
        slo = SLOLoader.load_slo(args.slo_file)
        if slo:
            print(f"Loaded SLO configuration: {args.slo_file}")
        else:
            print(f"Warning: Failed to load SLO configuration file or file format error: {args.slo_file}")
    
    config = BenchmarkConfig(
        dataset_path=args.dataset,
        endpoint_url=args.endpoint,
        num_samples=final_num_samples,
        concurrency_levels=concurrency_levels,
        mode=mode,
        max_input_length=args.max_input_length,
        max_output_tokens=args.max_output_tokens,
        min_output_tokens=args.min_output_tokens,
        model_name=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        shuffle_round2=not args.no_shuffle_round2,
        slo_file=args.slo_file,
        output_dir=args.output_dir,
        prometheus_url=args.prometheus_url,
        prometheus_metrics=args.prometheus_metrics,
        save_requests=args.save_requests,
        reset_cache_url=args.reset_cache_url,
        reset_cache_between_rounds=args.reset_cache_between_rounds,
        reset_cache_between_concurrency=args.reset_cache_between_concurrency,
        debug=args.debug,
        debug_verbose=args.debug_verbose,
        debug_log_dir=args.debug_log_dir,
        max_context_tokens=args.max_context_tokens
    )
    
    print("="*60)
    print("Dual Round Benchmark Tool - Multi-Concurrency Test")
    print("="*60)
    print(f"Dataset: {config.dataset_path}")
    print(f"Endpoint: {config.endpoint_url}")
    print(f"Samples: {config.num_samples}")
    print(f"Concurrency: {', '.join(map(str, config.concurrency_levels))}")
    print(f"Max input length: {config.max_input_length if config.max_input_length else 'Unlimited'}")
    print(f"Max output tokens: {config.max_output_tokens if config.max_output_tokens else 'Unlimited'}")
    print(f"Model: {config.model_name}")
    print(f"Shuffle round 2: {'Yes' if config.shuffle_round2 else 'No'}")
    if slo:
        print(f"SLO configuration: Loaded")
        if slo.ttft_ms:
            print(f"  - TTFT max: {slo.ttft_ms} ms")
        if slo.itl_ms:
            print(f"  - ITL max: {slo.itl_ms} ms")
        if slo.latency_ms:
            print(f"  - Latency max: {slo.latency_ms} ms")
        if slo.output_token_throughput:
            print(f"  - Output throughput min: {slo.output_token_throughput} tokens/sec")
    if config.prometheus_url:
        print(f"Prometheus URL: {config.prometheus_url}")
        if config.prometheus_metrics:
            print(f"Prometheus metrics: {', '.join(config.prometheus_metrics)}")
    if config.reset_cache_url:
        print(f"KVCache reset: Enabled")
        print(f"  - Reset endpoint: {config.reset_cache_url}")
        if config.reset_cache_between_rounds:
            print(f"  - Reset between rounds: Yes")
        if config.reset_cache_between_concurrency:
            print(f"  - Reset between concurrency levels: Yes")
    print("="*60)
    
    async def run_benchmark():
        mock_task = None
        if args.mock_server:
            from llm_mocker import run_server_until_cancelled
            shutdown_event = asyncio.Event()
            
            mock_task = asyncio.create_task(
                run_server_until_cancelled(args.mock_host, args.mock_port, shutdown_event)
            )
            print(f"Mock server started: http://{args.mock_host}:{args.mock_port}")
            await asyncio.sleep(2)
        
        runner = BenchmarkRunner(config, slo)
        all_raw_results = await runner.run()
        
        if mock_task:
            shutdown_event.set()
            await mock_task
        
        return all_raw_results
    
    all_raw_results = asyncio.run(run_benchmark())
    
    all_results = {}
    all_metrics_only = {}
    
    for concurrency, (round1_results, round2_results, round1_prom_metrics, round2_prom_metrics) in all_raw_results.items():
        round1_metrics = MetricsAnalyzer.analyze_round(1, round1_results, round1_prom_metrics)
        MetricsAnalyzer.print_metrics(concurrency, round1_metrics)

        # Multi-turn mode only runs one round, don't print second round
        if mode == BenchmarkMode.DUAL_ROUND and round2_results:
            round2_metrics = MetricsAnalyzer.analyze_round(2, round2_results, round2_prom_metrics)
            MetricsAnalyzer.print_metrics(concurrency, round2_metrics)
        else:
            # Multi-turn mode creates empty round2_metrics
            round2_metrics = MetricsAnalyzer.analyze_round(2, [], {})

        all_results[concurrency] = (round1_metrics, round2_metrics, round1_results, round2_results)
        all_metrics_only[concurrency] = (round1_metrics, round2_metrics)
    
    MetricsAnalyzer.save_results(all_results, args.output)
    
    MetricsAnalyzer.save_csv_results(
        config,
        all_metrics_only,
        slo,
        config.output_dir
    )
    
    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
