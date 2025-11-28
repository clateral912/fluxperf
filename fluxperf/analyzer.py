import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    BenchmarkConfig,
    RequestMetrics,
    RoundMetrics,
    SLOConstraints,
)


class MetricsAnalyzer:
    @staticmethod
    def calculate_percentile(values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = math.ceil(len(sorted_values) * percentile / 100) - 1
        return sorted_values[max(0, min(index, len(sorted_values) - 1))]

    @staticmethod
    def analyze_round(round_num: int, results: List[RequestMetrics], prometheus_metrics: Dict[str, List[float]] = None, 
                     stage_name: Optional[str] = None, concurrency: Optional[int] = None) -> RoundMetrics:
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        prom_stats = {}
        if prometheus_metrics:
            for metric_name, values in prometheus_metrics.items():
                if values:
                    prom_stats[metric_name] = {
                        'avg': mean(values),
                        'p50': median(values),
                        'p90': MetricsAnalyzer.calculate_percentile(values, 90),
                        'p99': MetricsAnalyzer.calculate_percentile(values, 99),
                        'min': min(values),
                        'max': max(values),
                        'stddev': stdev(values) if len(values) > 1 else 0.0
                    }

        if not successful:
            return RoundMetrics(
                round_num=round_num,
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(failed),
                avg_ttft=0, p50_ttft=0, p90_ttft=0, p95_ttft=0, p99_ttft=0, min_ttft=0, max_ttft=0, stddev_ttft=0,
                avg_itl=0, p50_itl=0, p90_itl=0, p95_itl=0, p99_itl=0, min_itl=0, max_itl=0, stddev_itl=0,
                avg_latency=0, p50_latency=0, p90_latency=0, p95_latency=0, p99_latency=0, min_latency=0, max_latency=0, stddev_latency=0,
                total_throughput=0,
                request_throughput=0,
                avg_input_tokens=0, p50_input_tokens=0, p90_input_tokens=0, p99_input_tokens=0, min_input_tokens=0, max_input_tokens=0, stddev_input_tokens=0,
                avg_output_tokens=0, p50_output_tokens=0, p90_output_tokens=0, p99_output_tokens=0, min_output_tokens=0, max_output_tokens=0, stddev_output_tokens=0,
                duration=0,
                goodput_requests=0,
                goodput_tokens=0,
                goodput_request_rate=0.0,
                goodput_token_rate=0.0,
                prometheus_metrics=prom_stats,
                stage_name=stage_name,
                concurrency=concurrency
            )

        ttft_values = [r.time_to_first_token * 1000 for r in successful if r.time_to_first_token > 0]
        latency_values = [r.total_latency * 1000 for r in successful]

        all_itls = []
        for r in successful:
            all_itls.extend(r.inter_token_latencies)

        start_timestamps = [r.start_timestamp for r in results]
        end_timestamps = [r.end_timestamp for r in successful if r.end_timestamp > 0]

        if start_timestamps and end_timestamps:
            duration = max(end_timestamps) - min(start_timestamps)
        else:
            duration = 0

        total_output_tokens = sum(r.output_tokens for r in successful)
        total_throughput = total_output_tokens / duration if duration > 0 else 0

        request_throughput = len(successful) / duration if duration > 0 else 0

        goodput_results = [r for r in successful if r.meets_slo]
        goodput_requests = len(goodput_results)
        goodput_tokens = sum(r.output_tokens for r in goodput_results)
        goodput_request_rate = (goodput_requests / len(successful) * 100) if successful else 0
        goodput_token_rate = (goodput_tokens / total_output_tokens * 100) if total_output_tokens > 0 else 0

        input_token_values = [r.input_tokens for r in successful]
        output_token_values = [r.output_tokens for r in successful]

        return RoundMetrics(
            round_num=round_num,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            stage_name=stage_name,
            concurrency=concurrency,
            avg_ttft=mean(ttft_values) if ttft_values else 0,
            p50_ttft=median(ttft_values) if ttft_values else 0,
            p90_ttft=MetricsAnalyzer.calculate_percentile(ttft_values, 90),
            p95_ttft=MetricsAnalyzer.calculate_percentile(ttft_values, 95),
            p99_ttft=MetricsAnalyzer.calculate_percentile(ttft_values, 99),
            min_ttft=min(ttft_values) if ttft_values else 0,
            max_ttft=max(ttft_values) if ttft_values else 0,
            stddev_ttft=stdev(ttft_values) if len(ttft_values) > 1 else 0,
            avg_itl=mean(all_itls) if all_itls else 0,
            p50_itl=median(all_itls) if all_itls else 0,
            p90_itl=MetricsAnalyzer.calculate_percentile(all_itls, 90),
            p95_itl=MetricsAnalyzer.calculate_percentile(all_itls, 95),
            p99_itl=MetricsAnalyzer.calculate_percentile(all_itls, 99),
            min_itl=min(all_itls) if all_itls else 0,
            max_itl=max(all_itls) if all_itls else 0,
            stddev_itl=stdev(all_itls) if len(all_itls) > 1 else 0,
            avg_latency=mean(latency_values) if latency_values else 0,
            p50_latency=median(latency_values) if latency_values else 0,
            p90_latency=MetricsAnalyzer.calculate_percentile(latency_values, 90),
            p95_latency=MetricsAnalyzer.calculate_percentile(latency_values, 95),
            p99_latency=MetricsAnalyzer.calculate_percentile(latency_values, 99),
            min_latency=min(latency_values) if latency_values else 0,
            max_latency=max(latency_values) if latency_values else 0,
            stddev_latency=stdev(latency_values) if len(latency_values) > 1 else 0,
            total_throughput=total_throughput,
            request_throughput=request_throughput,
            avg_input_tokens=mean(input_token_values) if input_token_values else 0,
            p50_input_tokens=median(input_token_values) if input_token_values else 0,
            p90_input_tokens=MetricsAnalyzer.calculate_percentile(input_token_values, 90),
            p99_input_tokens=MetricsAnalyzer.calculate_percentile(input_token_values, 99),
            min_input_tokens=min(input_token_values) if input_token_values else 0,
            max_input_tokens=max(input_token_values) if input_token_values else 0,
            stddev_input_tokens=stdev(input_token_values) if len(input_token_values) > 1 else 0,
            avg_output_tokens=mean(output_token_values) if output_token_values else 0,
            p50_output_tokens=median(output_token_values) if output_token_values else 0,
            p90_output_tokens=MetricsAnalyzer.calculate_percentile(output_token_values, 90),
            p99_output_tokens=MetricsAnalyzer.calculate_percentile(output_token_values, 99),
            min_output_tokens=min(output_token_values) if output_token_values else 0,
            max_output_tokens=max(output_token_values) if output_token_values else 0,
            stddev_output_tokens=stdev(output_token_values) if len(output_token_values) > 1 else 0,
            duration=duration,
            goodput_requests=goodput_requests,
            goodput_tokens=goodput_tokens,
            goodput_request_rate=goodput_request_rate,
            goodput_token_rate=goodput_token_rate,
            prometheus_metrics=prom_stats
        )

    @staticmethod
    def _truncate_text(text: str, max_width: int) -> str:
        if len(text) <= max_width:
            return text
        return text[:max_width - 3] + "..."

    @staticmethod
    def print_metrics(concurrency: int, metrics: RoundMetrics):
        table_data = []

        table_data.append([
            "Time to first token (ms)",
            f"{metrics.avg_ttft:.2f}",
            f"{metrics.p99_ttft:.2f}",
            f"{metrics.p90_ttft:.2f}",
            f"{metrics.p50_ttft:.2f}",
            f"{metrics.min_ttft:.2f}",
            f"{metrics.max_ttft:.2f}",
            f"{metrics.stddev_ttft:.2f}"
        ])

        table_data.append([
            "Inter token latency (ms)",
            f"{metrics.avg_itl:.2f}",
            f"{metrics.p99_itl:.2f}",
            f"{metrics.p90_itl:.2f}",
            f"{metrics.p50_itl:.2f}",
            f"{metrics.min_itl:.2f}",
            f"{metrics.max_itl:.2f}",
            f"{metrics.stddev_itl:.2f}"
        ])

        table_data.append([
            "Request latency (ms)",
            f"{metrics.avg_latency:.2f}",
            f"{metrics.p99_latency:.2f}",
            f"{metrics.p90_latency:.2f}",
            f"{metrics.p50_latency:.2f}",
            f"{metrics.min_latency:.2f}",
            f"{metrics.max_latency:.2f}",
            f"{metrics.stddev_latency:.2f}"
        ])

        table_data.append([
            "Input sequence length",
            f"{metrics.avg_input_tokens:.2f}",
            f"{metrics.p99_input_tokens:.2f}",
            f"{metrics.p90_input_tokens:.2f}",
            f"{metrics.p50_input_tokens:.2f}",
            f"{metrics.min_input_tokens:.0f}",
            f"{metrics.max_input_tokens:.0f}",
            f"{metrics.stddev_input_tokens:.2f}"
        ])

        table_data.append([
            "Output sequence length",
            f"{metrics.avg_output_tokens:.2f}",
            f"{metrics.p99_output_tokens:.2f}",
            f"{metrics.p90_output_tokens:.2f}",
            f"{metrics.p50_output_tokens:.2f}",
            f"{metrics.min_output_tokens:.0f}",
            f"{metrics.max_output_tokens:.0f}",
            f"{metrics.stddev_output_tokens:.2f}"
        ])

        table_data.append([
            "Output token throughput (per sec)",
            f"{metrics.total_throughput:.2f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Request throughput (per sec)",
            f"{metrics.request_throughput:.2f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Goodput requests",
            f"{metrics.goodput_requests}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Goodput requests rate (%)",
            f"{metrics.goodput_request_rate:.2f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Goodput tokens",
            f"{metrics.goodput_tokens}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Goodput tokens rate (%)",
            f"{metrics.goodput_token_rate:.2f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        if metrics.prometheus_metrics:
            for metric_name, stats in sorted(metrics.prometheus_metrics.items()):
                table_data.append([
                    MetricsAnalyzer._truncate_text(metric_name, 35),
                    f"{stats.get('avg', 0):.4f}",
                    f"{stats.get('p99', 0):.4f}",
                    f"{stats.get('p90', 0):.4f}",
                    f"{stats.get('p50', 0):.4f}",
                    f"{stats.get('min', 0):.4f}",
                    f"{stats.get('max', 0):.4f}",
                    f"{stats.get('stddev', 0):.4f}"
                ])

        headers = ["Statistic", "avg", "p99", "p90", "p50", "min", "max", "stddev"]
        col_widths = [35, 8, 8, 8, 8, 8, 8, 8]

        total_width = sum(col_widths) + len(col_widths) * 3 - 1

        if metrics.stage_name:
            title = f"{metrics.stage_name} - Round {metrics.round_num}"
        else:
            title = f"Dual Round Benchmarker | LLM Metrics (Concurrency: {concurrency}, Round: {metrics.round_num})"
        print(f"\n{title.center(total_width)}")

        top_border = "┏" + "┳".join("━" * (w + 2) for w in col_widths) + "┓"
        print(top_border)

        header_row = "┃ " + " ┃ ".join(
            headers[i].rjust(col_widths[i]) for i in range(len(headers))
        ) + " ┃"
        print(header_row)

        header_separator = "┡" + "╇".join("━" * (w + 2) for w in col_widths) + "┩"
        print(header_separator)

        for row in table_data:
            truncated_row = [MetricsAnalyzer._truncate_text(str(cell), col_widths[i]) for i, cell in enumerate(row)]
            data_row = "│ " + " │ ".join(
                truncated_row[i].rjust(col_widths[i]) for i in range(len(truncated_row))
            ) + " │"
            print(data_row)

        bottom_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
        print(bottom_border)

        print(f"\nConcurrency: {concurrency} | Round {metrics.round_num} | Total requests: {metrics.total_requests} | Successful: {metrics.successful_requests} | Failed: {metrics.failed_requests} | Duration: {metrics.duration:.2f}s")
        if metrics.goodput_requests > 0 or metrics.goodput_tokens > 0:
            print(f"Goodput: {metrics.goodput_requests} requests ({metrics.goodput_request_rate:.1f}%) | {metrics.goodput_tokens} tokens ({metrics.goodput_token_rate:.1f}%)")

    @staticmethod
    def serialize_results(
        all_results: Dict[int, Tuple[RoundMetrics, RoundMetrics, List[RequestMetrics], List[RequestMetrics]]]
    ) -> Dict[str, Any]:
        data = {}

        for concurrency, (round1_metrics, round2_metrics, round1_results, round2_results) in all_results.items():
            data[f"concurrency_{concurrency}"] = {
                "summary": {
                    "round_1": {
                        "total_requests": round1_metrics.total_requests,
                        "successful_requests": round1_metrics.successful_requests,
                        "failed_requests": round1_metrics.failed_requests,
                        "duration_seconds": round1_metrics.duration,
                        "ttft": {
                            "avg_ms": round1_metrics.avg_ttft,
                            "p50_ms": round1_metrics.p50_ttft,
                            "p95_ms": round1_metrics.p95_ttft,
                            "p99_ms": round1_metrics.p99_ttft
                        },
                        "itl": {
                            "avg_ms": round1_metrics.avg_itl,
                            "p50_ms": round1_metrics.p50_itl,
                            "p95_ms": round1_metrics.p95_itl,
                            "p99_ms": round1_metrics.p99_itl
                        },
                        "latency": {
                            "avg_ms": round1_metrics.avg_latency,
                            "p50_ms": round1_metrics.p50_latency
                        },
                        "throughput": {
                            "tokens_per_sec": round1_metrics.total_throughput,
                            "requests_per_sec": round1_metrics.request_throughput
                        },
                        "goodput": {
                            "requests": round1_metrics.goodput_requests,
                            "tokens": round1_metrics.goodput_tokens,
                            "request_rate_percent": round1_metrics.goodput_request_rate,
                            "token_rate_percent": round1_metrics.goodput_token_rate
                        },
                        "tokens": {
                            "avg_input": round1_metrics.avg_input_tokens,
                            "avg_output": round1_metrics.avg_output_tokens
                        }
                    },
                    "round_2": {
                        "total_requests": round2_metrics.total_requests,
                        "successful_requests": round2_metrics.successful_requests,
                        "failed_requests": round2_metrics.failed_requests,
                        "duration_seconds": round2_metrics.duration,
                        "ttft": {
                            "avg_ms": round2_metrics.avg_ttft,
                            "p50_ms": round2_metrics.p50_ttft,
                            "p95_ms": round2_metrics.p95_ttft,
                            "p99_ms": round2_metrics.p99_ttft
                        },
                        "itl": {
                            "avg_ms": round2_metrics.avg_itl,
                            "p50_ms": round2_metrics.p50_itl,
                            "p95_ms": round2_metrics.p95_itl,
                            "p99_ms": round2_metrics.p99_itl
                        },
                        "latency": {
                            "avg_ms": round2_metrics.avg_latency,
                            "p50_ms": round2_metrics.p50_latency
                        },
                        "throughput": {
                            "tokens_per_sec": round2_metrics.total_throughput,
                            "requests_per_sec": round2_metrics.request_throughput
                        },
                        "goodput": {
                            "requests": round2_metrics.goodput_requests,
                            "tokens": round2_metrics.goodput_tokens,
                            "request_rate_percent": round2_metrics.goodput_request_rate,
                            "token_rate_percent": round2_metrics.goodput_token_rate
                        },
                        "tokens": {
                            "avg_input": round2_metrics.avg_input_tokens,
                            "avg_output": round2_metrics.avg_output_tokens
                        }
                    }
                },
                "detailed_results": {
                    "round_1": [
                        {
                            "request_id": r.request_id,
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                            "ttft_ms": r.time_to_first_token * 1000,
                            "avg_itl_ms": mean(r.inter_token_latencies) if r.inter_token_latencies else 0,
                            "total_latency_ms": r.total_latency * 1000,
                            "throughput": r.throughput,
                            "meets_slo": r.meets_slo,
                            "error": r.error
                        }
                        for r in round1_results
                    ],
                    "round_2": [
                        {
                            "request_id": r.request_id,
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                            "ttft_ms": r.time_to_first_token * 1000,
                            "avg_itl_ms": mean(r.inter_token_latencies) if r.inter_token_latencies else 0,
                            "total_latency_ms": r.total_latency * 1000,
                            "throughput": r.throughput,
                            "meets_slo": r.meets_slo,
                            "error": r.error
                        }
                        for r in round2_results
                    ]
                }
            }

        return data

    @staticmethod
    def save_results(
        all_results: Dict[int, Tuple[RoundMetrics, RoundMetrics, List[RequestMetrics], List[RequestMetrics]]],
        output_path: Path
    ):
        data = MetricsAnalyzer.serialize_results(all_results)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nJSON results saved to: {output_path}")

    @staticmethod
    def save_csv_results(
        config: BenchmarkConfig,
        all_results: Dict[int, Tuple[RoundMetrics, RoundMetrics]],
        slo: Optional[SLOConstraints],
        output_dir: Path
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"benchmark_{timestamp}.csv"
        csv_path = output_dir / csv_filename
        
        params_filename = f"benchmark_{timestamp}_params.txt"
        params_path = output_dir / params_filename
        
        with open(params_path, 'w', encoding='utf-8') as f:
            f.write("=== Benchmark Parameters ===\n")
            f.write(f"Dataset: {config.dataset_path}\n")
            f.write(f"Endpoint: {config.endpoint_url}\n")
            f.write(f"Samples: {config.num_samples}\n")
            f.write(f"Concurrency: {', '.join(map(str, config.concurrency_levels))}\n")
            f.write(f"Max input length: {config.max_input_length if config.max_input_length else 'Unlimited'}\n")
            f.write(f"Model: {config.model_name}\n")
            f.write(f"Timeout: {config.timeout}s\n")
            f.write(f"Shuffle round 2: {'Yes' if config.shuffle_round2 else 'No'}\n")
            
            if slo:
                f.write(f"\n=== SLO Constraints ===\n")
                if slo.ttft_ms:
                    f.write(f"TTFT max: {slo.ttft_ms} ms\n")
                if slo.itl_ms:
                    f.write(f"ITL max: {slo.itl_ms} ms\n")
                if slo.latency_ms:
                    f.write(f"Latency max: {slo.latency_ms} ms\n")
                if slo.output_token_throughput:
                    f.write(f"Output throughput min: {slo.output_token_throughput} tokens/sec\n")
            
            f.write(f"\n=== Execution Time ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            concurrency_levels = sorted(all_results.keys())
            
            header = ['Metric/Round']
            for conc in concurrency_levels:
                header.extend([f'Concurrency{conc}-Round1', f'Concurrency{conc}-Round2'])
            writer.writerow(header)
            writer.writerow([])
            
            writer.writerow(['Basic Statistics'])
            
            row = ['Total requests']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([r1_metrics.total_requests, r2_metrics.total_requests])
            writer.writerow(row)
            
            row = ['Successful requests']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([r1_metrics.successful_requests, r2_metrics.successful_requests])
            writer.writerow(row)
            
            row = ['Failed requests']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([r1_metrics.failed_requests, r2_metrics.failed_requests])
            writer.writerow(row)
            
            row = ['Test duration (sec)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.duration:.2f}', f'{r2_metrics.duration:.2f}'])
            writer.writerow(row)
            writer.writerow([])
            
            writer.writerow(['TTFT (ms)'])
            row = ['Average']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.avg_ttft:.2f}', f'{r2_metrics.avg_ttft:.2f}'])
            writer.writerow(row)

            row = ['P50']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p50_ttft:.2f}', f'{r2_metrics.p50_ttft:.2f}'])
            writer.writerow(row)

            row = ['P90']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p90_ttft:.2f}', f'{r2_metrics.p90_ttft:.2f}'])
            writer.writerow(row)

            row = ['P95']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p95_ttft:.2f}', f'{r2_metrics.p95_ttft:.2f}'])
            writer.writerow(row)

            row = ['P99']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p99_ttft:.2f}', f'{r2_metrics.p99_ttft:.2f}'])
            writer.writerow(row)

            row = ['Min']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.min_ttft:.2f}', f'{r2_metrics.min_ttft:.2f}'])
            writer.writerow(row)

            row = ['Max']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.max_ttft:.2f}', f'{r2_metrics.max_ttft:.2f}'])
            writer.writerow(row)

            row = ['Std Dev']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.stddev_ttft:.2f}', f'{r2_metrics.stddev_ttft:.2f}'])
            writer.writerow(row)
            writer.writerow([])

            writer.writerow(['ITL (ms)'])
            row = ['Average']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.avg_itl:.2f}', f'{r2_metrics.avg_itl:.2f}'])
            writer.writerow(row)

            row = ['P50']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p50_itl:.2f}', f'{r2_metrics.p50_itl:.2f}'])
            writer.writerow(row)

            row = ['P90']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p90_itl:.2f}', f'{r2_metrics.p90_itl:.2f}'])
            writer.writerow(row)

            row = ['P95']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p95_itl:.2f}', f'{r2_metrics.p95_itl:.2f}'])
            writer.writerow(row)

            row = ['P99']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p99_itl:.2f}', f'{r2_metrics.p99_itl:.2f}'])
            writer.writerow(row)

            row = ['Min']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.min_itl:.2f}', f'{r2_metrics.min_itl:.2f}'])
            writer.writerow(row)

            row = ['Max']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.max_itl:.2f}', f'{r2_metrics.max_itl:.2f}'])
            writer.writerow(row)

            row = ['Std Dev']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.stddev_itl:.2f}', f'{r2_metrics.stddev_itl:.2f}'])
            writer.writerow(row)
            writer.writerow([])
            
            writer.writerow(['Throughput'])
            row = ['Token throughput (tokens/sec)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.total_throughput:.2f}', f'{r2_metrics.total_throughput:.2f}'])
            writer.writerow(row)
            
            row = ['Request throughput (requests/sec)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.request_throughput:.2f}', f'{r2_metrics.request_throughput:.2f}'])
            writer.writerow(row)
            writer.writerow([])
            
            writer.writerow(['Goodput'])
            row = ['Requests meeting SLO']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([r1_metrics.goodput_requests, r2_metrics.goodput_requests])
            writer.writerow(row)
            
            row = ['Request Goodput rate (%)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.goodput_request_rate:.2f}', f'{r2_metrics.goodput_request_rate:.2f}'])
            writer.writerow(row)
            
            row = ['Token Goodput rate (%)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.goodput_token_rate:.2f}', f'{r2_metrics.goodput_token_rate:.2f}'])
            writer.writerow(row)
            
            if config.prometheus_metrics:
                all_metric_names = set()
                for conc in concurrency_levels:
                    r1_metrics, r2_metrics = all_results[conc]
                    all_metric_names.update(r1_metrics.prometheus_metrics.keys())
                    all_metric_names.update(r2_metrics.prometheus_metrics.keys())

                if all_metric_names:
                    writer.writerow([])
                    writer.writerow(['Prometheus Metrics'])

                    for metric_name in sorted(all_metric_names):
                        writer.writerow([f'{metric_name}'])

                        row = ['Average']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('avg', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('avg', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['P50']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('p50', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('p50', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['P90']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('p90', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('p90', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['P99']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('p99', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('p99', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['Min']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('min', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('min', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['Max']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('max', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('max', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['Std Dev']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('stddev', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('stddev', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)
                        writer.writerow([])
        
        print(f"CSV results saved to: {csv_path}")
        print(f"Parameters saved to: {params_path}")
