"""Unit tests for _extract_vllm_latency_metrics method."""

import statistics
from unittest.mock import MagicMock, patch

import pytest

from amd_bench.core.analysis import BenchmarkAnalyzer
from amd_bench.schemas.benchmark import BenchmarkMetrics


class TestExtractVllmLatencyMetrics:
    """Test _extract_vllm_latency_metrics static method."""

    def test_extract_basic_latency_data(self) -> None:
        """Test extraction with basic required data."""
        data = {
            "avg_latency": 2.5,
            "latencies": [2.3, 2.5, 2.7, 2.4, 2.6],
            "percentiles": {"50": 2.5, "90": 2.7, "95": 2.8, "99": 2.9},
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert isinstance(result, BenchmarkMetrics)
        assert result.avg_latency == 2.5
        assert result.throughput == pytest.approx(0.4, abs=0.001)  # 1/2.5
        assert result.p50_latency == 2.5
        assert result.p90_latency == 2.7
        assert result.p95_latency == 2.8
        assert result.p99_latency == 2.9
        assert result.total_iterations == 5
        assert result.tokens_per_second == 0.0

    def test_extract_with_integer_percentile_keys(self) -> None:
        """Test extraction with integer keys in percentiles dictionary."""
        data = {
            "avg_latency": 1.8,
            "latencies": [1.7, 1.8, 1.9],
            "percentiles": {50: 1.8, 90: 1.9, 95: 1.95, 99: 2.0},  # Integer keys
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.avg_latency == 1.8
        assert result.p50_latency == 1.8
        assert result.p90_latency == 1.9
        assert result.p95_latency == 1.95
        assert result.p99_latency == 2.0
        assert result.throughput == pytest.approx(0.556, abs=0.001)

    def test_extract_with_mixed_percentile_keys(self) -> None:
        """Test extraction with mixed string and integer percentile keys."""
        data = {
            "avg_latency": 3.2,
            "latencies": [3.0, 3.1, 3.2, 3.3, 3.4],
            "percentiles": {"50": 3.2, 90: 3.3, "95": 3.35, 99: 3.4},  # Mixed keys
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.p50_latency == 3.2
        assert result.p90_latency == 3.3
        assert result.p95_latency == 3.35
        assert result.p99_latency == 3.4

    def test_extract_with_missing_percentiles(self) -> None:
        """Test extraction when percentiles dictionary is missing some keys."""
        data = {
            "avg_latency": 1.5,
            "latencies": [1.4, 1.5, 1.6],
            "percentiles": {"50": 1.5, "90": 1.6},  # Missing p95, p99
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.p50_latency == 1.5
        assert result.p90_latency == 1.6
        assert result.p95_latency == 0.0  # Default value
        assert result.p99_latency == 0.0  # Default value

    def test_extract_with_empty_percentiles(self) -> None:
        """Test extraction when percentiles dictionary is empty."""
        data = {"avg_latency": 2.0, "latencies": [1.9, 2.0, 2.1], "percentiles": {}}

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.p50_latency == 0.0
        assert result.p90_latency == 0.0
        assert result.p95_latency == 0.0
        assert result.p99_latency == 0.0

    def test_extract_with_missing_optional_fields(self) -> None:
        """Test extraction when optional fields are missing."""
        data = {
            "avg_latency": 1.2
            # Missing latencies and percentiles
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.avg_latency == 1.2
        assert result.throughput == pytest.approx(0.833, abs=0.001)
        assert result.latency_std == 0.0
        assert result.total_iterations == 1  # Default when no latencies
        assert result.p50_latency == 0.0
        assert result.p90_latency == 0.0

    def test_extract_with_empty_latencies_list(self) -> None:
        """Test extraction when latencies list is empty."""
        data = {"avg_latency": 2.8, "latencies": [], "percentiles": {"50": 2.8}}

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.latency_std == 0.0
        assert result.total_iterations == 1  # Default when empty list

    def test_extract_with_single_latency_value(self) -> None:
        """Test extraction when only one latency value exists."""
        data = {
            "avg_latency": 3.0,
            "latencies": [3.0],
            "percentiles": {"50": 3.0, "90": 3.0, "95": 3.0, "99": 3.0},
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.latency_std == 0.0  # No variation with single value
        assert result.total_iterations == 1

    def test_extract_latency_std_calculation(self) -> None:
        """Test that latency standard deviation is calculated correctly."""
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_std = statistics.stdev(latencies)

        data = {"avg_latency": 3.0, "latencies": latencies, "percentiles": {"50": 3.0}}

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.latency_std == pytest.approx(expected_std, abs=0.001)

    def test_extract_throughput_calculation(self) -> None:
        """Test that throughput is correctly calculated as per-request completion rate."""
        test_cases = [
            (1.0, 1.0),  # 1 second -> 1 req/s
            (2.0, 0.5),  # 2 seconds -> 0.5 req/s
            (0.5, 2.0),  # 0.5 seconds -> 2 req/s
            (0.25, 4.0),  # 0.25 seconds -> 4 req/s
        ]

        for avg_latency, expected_throughput in test_cases:
            data = {"avg_latency": avg_latency}
            result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)
            assert result.throughput == pytest.approx(expected_throughput, abs=0.001)

    @patch("statistics.stdev")
    @patch("amd_bench.core.analysis.logger")
    def test_extract_with_statistics_error(
        self, mock_logger: MagicMock, mock_stdev: MagicMock
    ) -> None:
        """Test handling of statistics calculation errors."""
        mock_stdev.side_effect = statistics.StatisticsError("Test error")

        data = {"avg_latency": 2.0, "latencies": [1.9, 2.0, 2.1], "percentiles": {"50": 2.0}}

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.latency_std == 0.0  # Should default to 0.0 on error
        mock_logger.warning.assert_called_with("Could not calculate latency standard deviation")

    def test_missing_avg_latency_raises_key_error(self) -> None:
        """Test that missing avg_latency raises KeyError."""
        data = {"latencies": [1.0, 2.0, 3.0], "percentiles": {"50": 2.0}}

        with pytest.raises(
            KeyError, match="Required field 'avg_latency' missing from benchmark data"
        ):
            BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

    def test_zero_avg_latency_raises_value_error(self) -> None:
        """Test that zero avg_latency raises ValueError."""
        data = {"avg_latency": 0.0, "latencies": [1.0, 2.0, 3.0]}

        with pytest.raises(ValueError, match="Invalid avg_latency value: 0.0. Must be positive."):
            BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

    def test_negative_avg_latency_raises_value_error(self) -> None:
        """Test that negative avg_latency raises ValueError."""
        data = {"avg_latency": -1.5, "latencies": [1.0, 2.0, 3.0]}

        with pytest.raises(ValueError, match="Invalid avg_latency value: -1.5. Must be positive."):
            BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

    def test_extract_with_very_small_latency(self) -> None:
        """Test extraction with very small latency values."""
        data = {
            "avg_latency": 0.001,  # 1ms
            "latencies": [0.0008, 0.001, 0.0012],
            "percentiles": {"50": 0.001, "90": 0.0012, "95": 0.0013, "99": 0.0015},
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.avg_latency == 0.001
        assert result.throughput == pytest.approx(1000.0, abs=0.001)  # 1000 req/s
        assert result.total_iterations == 3

    def test_extract_with_large_latency(self) -> None:
        """Test extraction with large latency values."""
        data = {
            "avg_latency": 30.0,  # 30 seconds
            "latencies": [28.0, 30.0, 32.0],
            "percentiles": {"50": 30.0, "90": 32.0, "95": 33.0, "99": 35.0},
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.avg_latency == 30.0
        assert result.throughput == pytest.approx(0.0333, abs=0.0001)  # ~0.033 req/s

    def test_extract_with_float_percentile_values(self) -> None:
        """Test extraction with float percentile values."""
        data = {
            "avg_latency": 2.5,
            "latencies": [2.3, 2.5, 2.7],
            "percentiles": {"50": 2.55, "90": 2.78, "95": 2.89, "99": 2.95},
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.p50_latency == pytest.approx(2.55, abs=0.001)
        assert result.p90_latency == pytest.approx(2.78, abs=0.001)
        assert result.p95_latency == pytest.approx(2.89, abs=0.001)
        assert result.p99_latency == pytest.approx(2.95, abs=0.001)

    def test_extract_tokens_per_second_always_zero(self) -> None:
        """Test that tokens_per_second is always 0.0 for latency metrics."""
        data = {"avg_latency": 1.5, "latencies": [1.4, 1.5, 1.6], "percentiles": {"50": 1.5}}

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.tokens_per_second == 0.0

    def test_extract_comprehensive_scenario(self) -> None:
        """Test extraction with comprehensive real-world scenario."""
        data = {
            "avg_latency": 4.125,
            "latencies": [3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5],
            "percentiles": {"50": 4.15, "90": 4.4, "95": 4.45, "99": 4.5},
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        # Verify all fields are populated correctly
        assert result.avg_latency == 4.125
        assert result.throughput == pytest.approx(0.2424, abs=0.0001)  # 1/4.125
        assert result.p50_latency == 4.15
        assert result.p90_latency == 4.4
        assert result.p95_latency == 4.45
        assert result.p99_latency == 4.5
        assert result.total_iterations == 8
        assert result.latency_std > 0  # Should have some variance
        assert result.tokens_per_second == 0.0

        # Verify latency_std calculation
        expected_std = statistics.stdev(data["latencies"])
        assert result.latency_std == pytest.approx(expected_std, abs=0.001)

    def test_per_request_completion_rate_documentation(self) -> None:
        """Test that demonstrates per-request completion rate vs system throughput."""
        # For a batch size of 8 with avg_latency of 2.5s per request
        data = {
            "avg_latency": 2.5,  # Individual request takes 2.5s to complete
            "latencies": [2.3, 2.5, 2.7, 2.4, 2.6],
            "percentiles": {"50": 2.5},
        }

        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        # Per-request completion rate (what this method calculates)
        per_request_rate = result.throughput
        assert per_request_rate == pytest.approx(0.4, abs=0.001)  # 0.4 req/s per experiment

        # System throughput would be calculated at experiment level as:
        # system_throughput = batch_size * per_request_rate
        # For batch_size = 8: system_throughput = 8 * 0.4 = 3.2 req/s
        # But this method only calculates the per-request rate

    @pytest.mark.parametrize("missing_field", ["latencies", "percentiles"])
    def test_extract_robust_handling_of_missing_optional_fields(self, missing_field: str) -> None:
        """Test that method handles missing optional fields gracefully."""
        data = {
            "avg_latency": 1.8,
            "latencies": [1.7, 1.8, 1.9],
            "percentiles": {"50": 1.8, "90": 1.9},
        }

        # Remove the specified field
        del data[missing_field]

        # Should not raise an exception
        result = BenchmarkAnalyzer._extract_vllm_latency_metrics(data)

        assert result.avg_latency == 1.8
        assert result.throughput == pytest.approx(0.556, abs=0.001)

        if missing_field == "latencies":
            assert result.latency_std == 0.0
            assert result.total_iterations == 1
        elif missing_field == "percentiles":
            assert result.p50_latency == 0.0
            assert result.p90_latency == 0.0
