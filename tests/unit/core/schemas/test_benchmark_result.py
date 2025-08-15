"""Unit tests for BenchmarkResult property methods"""

import pytest

from amd_bench.schemas.benchmark import BenchmarkMetrics, BenchmarkResult, ExperimentConfig


class TestBenchmarkResultProperties:
    """Test BenchmarkResult computed properties"""

    @pytest.fixture
    def sample_config(self):
        return ExperimentConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            benchmark_type="latency",
            batch_size=8,
            input_length=128,
            output_length=128,
            dtype="float16",
            memory_util=0.85,
            timestamp="20240812_143022",
        )

    @pytest.fixture
    def sample_metrics(self):
        return BenchmarkMetrics(
            avg_latency=2.0,
            latency_std=0.1,
            p50_latency=1.9,
            p90_latency=2.1,
            p95_latency=2.2,
            p99_latency=2.3,
            throughput=0.5,
            tokens_per_second=1000.0,
            total_iterations=10,
        )

    @pytest.fixture
    def sample_result(self, sample_config, sample_metrics):
        return BenchmarkResult(
            file="llama3_latency_bs8_in128_out128_float16_mem0.85_20240812_143022.json",
            experiment_id="test-experiment-id",
            config=sample_config,
            metrics=sample_metrics,
        )

    def test_model_short_name_extraction(self, sample_result):
        assert sample_result.model_short_name == "Llama-3.1-8B"

    def test_model_short_name_simple_name(self, sample_config, sample_metrics):
        config = sample_config.model_copy(update={"model": "simple-model"})
        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=config,
            metrics=sample_metrics,
        )
        assert result.model_short_name == "simple-model"

    def test_efficiency_score_calculation(self, sample_result):
        """Test efficiency score = throughput / avg_latency"""
        expected_efficiency = 0.5 / 2.0  # throughput / avg_latency
        assert abs(sample_result.efficiency_score - expected_efficiency) < 1e-6

    def test_efficiency_score_zero_latency(self, sample_config, sample_metrics):
        """Test efficiency score handles zero latency gracefully"""
        metrics_zero_latency = sample_metrics.model_copy(update={"avg_latency": 0.0})
        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=sample_config,
            metrics=metrics_zero_latency,
        )
        assert result.efficiency_score == 0.0

    def test_system_throughput_calculation(self, sample_result):
        """Test system throughput = batch_size * throughput"""
        expected_system_throughput = 8 * 0.5  # batch_size * throughput
        assert abs(sample_result.system_throughput - expected_system_throughput) < 1e-6

    def test_system_throughput_batch_size_one(self, sample_config, sample_metrics):
        """Test system throughput with batch size 1"""
        config_batch_one = sample_config.model_copy(update={"batch_size": 1})
        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=config_batch_one,
            metrics=sample_metrics,
        )
        expected_system_throughput = 1 * 0.5
        assert abs(result.system_throughput - expected_system_throughput) < 1e-6

    @pytest.mark.parametrize(
        ("batch_size", "throughput", "avg_latency", "expected_system_throughput"),
        [
            (1, 1.0, 1.0, 1.0),
            (4, 0.8, 1.25, 3.2),
            (16, 0.25, 4.0, 4.0),
            (32, 0.1, 10.0, 3.2),
        ],
    )
    def test_system_throughput_various_scenarios(
        self, sample_config, batch_size, throughput, avg_latency, expected_system_throughput
    ):
        """Test system throughput calculation with various realistic scenarios"""
        config = sample_config.model_copy(update={"batch_size": batch_size})
        metrics = BenchmarkMetrics(
            avg_latency=avg_latency,
            latency_std=0.1,
            p50_latency=avg_latency * 0.95,
            p90_latency=avg_latency * 1.05,
            p95_latency=avg_latency * 1.1,
            p99_latency=avg_latency * 1.15,
            throughput=throughput,
            tokens_per_second=1000.0,
            total_iterations=10,
        )

        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=config,
            metrics=metrics,
        )

        assert abs(result.system_throughput - expected_system_throughput) < 1e-6
