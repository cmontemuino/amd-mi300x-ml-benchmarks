"""Unit tests for the BatchEfficiencyAnalyzer class"""

from typing import List

import pytest

from amd_bench.core.analysis import BatchEfficiencyAnalyzer
from amd_bench.schemas.benchmark import BenchmarkMetrics, BenchmarkResult, ExperimentConfig


@pytest.fixture
def mock_benchmark_results() -> List[BenchmarkResult]:
    """
    Creates a list of mock BenchmarkResult objects for testing.
    - bs=1: baseline (throughput=100)
    - bs=2: 90% efficiency (180 system throughput vs 200 theoretical)
    - bs=4: 100% efficiency (400 system throughput vs 400 theoretical)
    - bs=8: 110% efficiency (880 system throughput vs 800 theoretical)
    """
    results = []
    # Throughput is req/sec for a single request/batch.
    # system_throughput is bs * throughput.
    test_data = [
        # Baseline
        {"bs": 1, "throughput": 100.0, "avg_latency": 0.01},
        # 90% efficient
        {"bs": 2, "throughput": 90.0, "avg_latency": 1 / 90.0},
        # 100% efficient
        {"bs": 4, "throughput": 100.0, "avg_latency": 1 / 100.0},
        # 110% efficient
        {"bs": 8, "throughput": 110.0, "avg_latency": 1 / 110.0},
    ]

    for data in test_data:
        config = ExperimentConfig(
            model="test-model",
            benchmark_type="latency",
            batch_size=data["bs"],
            input_length=128,
            output_length=128,
            dtype="float16",
            memory_util=0.8,
            timestamp="20250101_120000",
        )
        metrics = BenchmarkMetrics(
            avg_latency=data["avg_latency"],
            throughput=data["throughput"],
            total_iterations=100,
        )
        result = BenchmarkResult(
            file=f"file_bs{data['bs']}.json",
            experiment_id=f"exp_bs{data['bs']}",
            config=config,
            metrics=metrics,
        )
        results.append(result)
    return results


class TestBatchEfficiencyAnalyzer:
    """Tests for the BatchEfficiencyAnalyzer class."""

    def test_initialization(self, mock_benchmark_results):
        """Test that the analyzer initializes and groups results correctly."""
        analyzer = BatchEfficiencyAnalyzer(mock_benchmark_results)
        assert len(analyzer.by_batch_size) == 4
        assert 1 in analyzer.by_batch_size
        assert 8 in analyzer.by_batch_size
        assert len(analyzer.by_batch_size[1]) == 1

    def test_calculate_scaling_efficiency_default_baseline(self, mock_benchmark_results):
        """Test scaling efficiency calculation with the default baseline (bs=1)."""
        analyzer = BatchEfficiencyAnalyzer(mock_benchmark_results)
        efficiencies = analyzer.calculate_scaling_efficiency()

        # bs=1: 100 / (1*100) = 1.0
        # bs=2: (2*90) / (2*100) = 0.9
        # bs=4: (4*100) / (4*100) = 1.0
        # bs=8: (8*110) / (8*100) = 1.1
        assert efficiencies[1] == pytest.approx(1.0)
        assert efficiencies[2] == pytest.approx(0.9)
        assert efficiencies[4] == pytest.approx(1.0)
        assert efficiencies[8] == pytest.approx(1.1)

    def test_calculate_scaling_efficiency_custom_baseline(self, mock_benchmark_results):
        """Test scaling efficiency calculation with a custom baseline."""
        analyzer = BatchEfficiencyAnalyzer(mock_benchmark_results)
        # Use bs=2 as the baseline (throughput=90)
        efficiencies = analyzer.calculate_scaling_efficiency(baseline_batch_size=2)

        # bs=1: 100 / (1*90) = 1.111...
        # bs=2: (2*90) / (2*90) = 1.0
        # bs=4: (4*100) / (4*90) = 1.111...
        # bs=8: (8*110) / (8*90) = 1.222...
        assert efficiencies[1] == pytest.approx(100.0 / 90.0)
        assert efficiencies[2] == pytest.approx(1.0)
        assert efficiencies[4] == pytest.approx(400.0 / 360.0)
        assert efficiencies[8] == pytest.approx(880.0 / 720.0)

    def test_calculate_scaling_efficiency_missing_baseline(self, mock_benchmark_results):
        """Test that a ValueError is raised if the baseline is not found."""
        analyzer = BatchEfficiencyAnalyzer(mock_benchmark_results)
        with pytest.raises(ValueError, match="No data for baseline batch size 5"):
            analyzer.calculate_scaling_efficiency(baseline_batch_size=5)

    def test_get_scaling_grades(self, mock_benchmark_results):
        """Test the generation of human-readable scaling grades."""
        analyzer = BatchEfficiencyAnalyzer(mock_benchmark_results)
        grades = analyzer.get_scaling_grades()

        # Based on efficiencies: 1.0, 0.9, 1.0, 1.1
        assert grades[1] == "A (Very Good)"
        assert grades[2] == "B (Good)"
        assert grades[4] == "A (Very Good)"
        assert grades[8] == "A+ (Excellent)"
