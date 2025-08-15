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

    def test_batch_efficiency_ratio_perfect_scaling_scenario(self, sample_config):
        """Test with metrics that represent good batch scaling"""
        perfect_metrics = BenchmarkMetrics(
            avg_latency=2.0,  # Same latency as single request
            latency_std=0.05,  # Low variance
            p50_latency=1.9,
            p90_latency=2.1,
            p95_latency=2.2,
            p99_latency=2.3,
            throughput=4.0,  # Perfect throughput scaling (8 * 0.5)
            tokens_per_second=1000.0,
            total_iterations=10,
        )

        # High memory utilization for better efficiency
        config = sample_config.model_copy(update={"memory_util": 0.95})
        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=config,
            metrics=perfect_metrics,
        )

        # Should have high efficiency ratio due to perfect throughput scaling
        assert result.batch_efficiency_ratio > 0.8
        assert result.throughput_scaling_efficiency == pytest.approx(1.0, rel=1e-3)

    def test_batch_efficiency_ratio_degraded_performance(self, sample_config):
        """Test with metrics showing poor batch scaling"""
        degraded_metrics = BenchmarkMetrics(
            avg_latency=8.0,  # Much higher latency
            latency_std=1.0,  # High variance
            p50_latency=7.5,
            p90_latency=9.0,
            p95_latency=9.5,
            p99_latency=10.0,
            throughput=0.1,  # Poor throughput
            tokens_per_second=500.0,
            total_iterations=10,
        )

        # Low memory utilization
        config = sample_config.model_copy(update={"memory_util": 0.5})
        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=config,
            metrics=degraded_metrics,
        )

        # Should have low efficiency ratio
        assert result.batch_efficiency_ratio < 0.5
        assert result.throughput_scaling_efficiency < 0.2

    def test_batch_scaling_grade(self, sample_config, sample_metrics):
        """Test the human-readable grading system"""
        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=sample_config,
            metrics=sample_metrics,
        )

        grade = result.batch_scaling_grade
        assert grade in [
            "A+ (Excellent)",
            "A (Very Good)",
            "B (Good)",
            "C (Fair)",
            "D (Poor)",
            "F (Very Poor)",
        ]

    def test_throughput_scaling_efficiency(self, sample_result):
        """Test pure throughput scaling efficiency calculation"""
        # For batch_size=8, avg_latency=2.0, throughput=0.5
        # theoretical_max = 8 * (1.0 / 2.0) = 4.0
        # efficiency = 0.5 / 4.0 = 0.125
        expected_efficiency = 0.125
        assert sample_result.throughput_scaling_efficiency == pytest.approx(
            expected_efficiency, rel=1e-3
        )

    def test_latency_scaling_efficiency(self, sample_result):
        """Test latency scaling efficiency calculation"""
        # Should return a value between 0 and 1
        efficiency = sample_result.latency_scaling_efficiency
        assert 0.0 <= efficiency <= 1.0

    def test_resource_utilization_score(self, sample_result):
        """Test resource utilization score calculation"""
        # Should return a value between 0 and 1
        score = sample_result.resource_utilization_score
        assert 0.0 <= score <= 1.0

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

    def test_batch_efficiency_edge_cases(self, sample_config):
        """Test edge cases for batch efficiency calculation"""

        # Test with zero latency standard deviation
        metrics_no_variance = BenchmarkMetrics(
            avg_latency=2.0,
            latency_std=0.0,  # No variance
            p50_latency=2.0,
            p90_latency=2.0,
            p95_latency=2.0,
            p99_latency=2.0,
            throughput=0.5,
            tokens_per_second=1000.0,
            total_iterations=10,
        )

        result = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=sample_config,
            metrics=metrics_no_variance,
        )

        # Should handle zero variance gracefully
        assert 0.0 <= result.batch_efficiency_ratio <= 1.0

        # Test with very high memory utilization
        config_high_mem = sample_config.model_copy(update={"memory_util": 1.0})
        result_high_mem = BenchmarkResult(
            file="test.json",
            experiment_id="test-id",
            config=config_high_mem,
            metrics=metrics_no_variance,
        )

        # Should still be bounded
        assert 0.0 <= result_high_mem.batch_efficiency_ratio <= 1.0
