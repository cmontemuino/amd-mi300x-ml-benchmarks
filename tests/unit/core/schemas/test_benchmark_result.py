"""Unit tests for BenchmarkResult property methods"""

import pytest

from amd_bench.schemas.benchmark import BenchmarkMetrics, BenchmarkResult, ExperimentConfig


class TestBenchmarkResultProperties:
    """Test BenchmarkResult computed properties"""

    @pytest.fixture
    def sample_config(self):
        """Create a sample experiment configuration"""
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
        """Create sample benchmark metrics"""
        return BenchmarkMetrics(
            avg_latency=2.0,  # seconds
            latency_std=0.1,
            p50_latency=1.9,
            p90_latency=2.1,
            p95_latency=2.2,
            p99_latency=2.3,
            throughput=0.5,  # requests/sec (per-request completion rate)
            tokens_per_second=1000.0,
            total_iterations=10,
        )

    @pytest.fixture
    def sample_result(self, sample_config, sample_metrics):
        """Create a sample benchmark result"""
        return BenchmarkResult(
            file="llama3_latency_bs8_in128_out128_float16_mem0.85_20240812_143022.json",
            experiment_id="test-experiment-id",
            config=sample_config,
            metrics=sample_metrics,
        )

    def test_model_short_name_extraction(self, sample_result):
        """Test model short name extraction from full model path"""
        assert sample_result.model_short_name == "Llama-3.1-8B"

    def test_model_short_name_simple_name(self, sample_config, sample_metrics):
        """Test model short name with simple model name"""
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

    def test_batch_efficiency_ratio_calculation(self, sample_result):
        """Test batch efficiency ratio calculation"""
        # Theoretical throughput = batch_size * (1 / avg_latency) = 8 * (1 / 2.0) = 4.0
        # Actual completion rate = throughput = 0.5
        # Efficiency ratio = 0.5 / 4.0 = 0.125
        expected_ratio = 0.5 / (8 * (1.0 / 2.0))
        assert abs(sample_result.batch_efficiency_ratio - expected_ratio) < 1e-6

    def test_batch_efficiency_ratio_single_batch(self, sample_config, sample_metrics):
        """Test batch efficiency ratio returns 1.0 for batch_size <= 1"""
        config_batch_one = sample_config.model_copy(update={"batch_size": 1})
        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=config_batch_one,
            metrics=sample_metrics,
        )
        assert result.batch_efficiency_ratio == 1.0

    def test_batch_efficiency_ratio_perfect_scaling(self, sample_config):
        """Test batch efficiency ratio with perfect scaling scenario"""
        # Perfect scaling: throughput = batch_size / avg_latency
        perfect_metrics = BenchmarkMetrics(
            avg_latency=2.0,
            latency_std=0.1,
            p50_latency=1.9,
            p90_latency=2.1,
            p95_latency=2.2,
            p99_latency=2.3,
            throughput=4.0,  # Perfect scaling: 8 / 2.0 = 4.0
            tokens_per_second=1000.0,
            total_iterations=10,
        )

        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=sample_config,
            metrics=perfect_metrics,
        )

        # With perfect scaling, efficiency ratio should be 1.0
        assert abs(result.batch_efficiency_ratio - 1.0) < 1e-6

    def test_batch_efficiency_ratio_degraded_performance(self, sample_config):
        """Test batch efficiency ratio with degraded performance"""
        # Degraded performance: lower throughput than theoretical
        degraded_metrics = BenchmarkMetrics(
            avg_latency=4.0,  # Higher latency due to queueing
            latency_std=0.2,
            p50_latency=3.8,
            p90_latency=4.2,
            p95_latency=4.4,
            p99_latency=4.6,
            throughput=0.25,  # Much lower than ideal (8/4 = 2.0)
            tokens_per_second=500.0,
            total_iterations=10,
        )

        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=sample_config,
            metrics=degraded_metrics,
        )

        # Efficiency ratio should be < 1.0 due to degraded performance
        # Expected: 0.25 / (8 * (1/4)) = 0.25 / 2.0 = 0.125
        expected_ratio = 0.25 / (8 * (1.0 / 4.0))
        assert abs(result.batch_efficiency_ratio - expected_ratio) < 1e-6
        assert result.batch_efficiency_ratio < 1.0

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
