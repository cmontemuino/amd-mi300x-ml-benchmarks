"""Unit tests for BenchmarkResult property methods"""

import pytest

from amd_bench.schemas.benchmark import BenchmarkMetrics, BenchmarkResult, ExperimentConfig


class TestBenchmarkResultIntegration:
    """Integration tests for BenchmarkResult with realistic data"""

    def test_realistic_latency_benchmark_scenario(self):
        """Test with realistic vLLM latency benchmark data"""
        config = ExperimentConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            benchmark_type="latency",
            batch_size=32,
            input_length=128,
            output_length=256,
            dtype="float16",
            memory_util=0.9,
            timestamp="20240812_143022",
        )

        # Realistic metrics showing batch processing degradation
        metrics = BenchmarkMetrics(
            avg_latency=5.1,  # Higher latency due to larger batch
            latency_std=0.8,
            p50_latency=4.8,
            p90_latency=6.2,
            p95_latency=6.8,
            p99_latency=7.5,
            throughput=0.196,  # Per-request completion rate
            tokens_per_second=2048.0,
            total_iterations=50,
        )

        result = BenchmarkResult(
            file="llama3_latency_bs32_in128_out256_float16_mem0.9_20240812_143022.json",
            experiment_id="llama3_latency_32_0.9_20240812_143022",
            config=config,
            metrics=metrics,
        )

        # Verify properties make sense
        assert result.model_short_name == "Llama-3.1-8B"
        assert result.efficiency_score == pytest.approx(0.196 / 5.1, rel=1e-3)
        assert result.system_throughput == pytest.approx(32 * 0.196, rel=1e-3)

        # Batch efficiency should be < 1.0 showing degradation
        theoretical_max = 32 * (1.0 / 5.1)
        expected_efficiency = 0.196 / theoretical_max
        assert result.batch_efficiency_ratio == pytest.approx(expected_efficiency, rel=1e-3)
        assert result.batch_efficiency_ratio < 1.0  # Shows degraded performance
