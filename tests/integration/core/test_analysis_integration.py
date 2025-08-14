"""Integration tests for BenchmarkAnalyzer"""

import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest

from amd_bench.core.analysis import BenchmarkAnalyzer
from amd_bench.schemas.benchmark import AnalysisConfig


class TestBenchmarkAnalyzerIntegration:
    """Integration tests for BenchmarkAnalyzer with real file scenarios"""

    @pytest.fixture
    def real_config(self) -> Generator[AnalysisConfig, Any, None]:
        """Real configuration for integration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"

            input_dir.mkdir()

            # Create containerized subdirectory
            containerized_dir = input_dir / "containerized"
            containerized_dir.mkdir()

            # Create test files in containerized directory
            (
                containerized_dir
                / "llama3_latency_bs32_in128_out256_float16_mem0.9_20240812_143022.json"
            ).write_text(
                '{"avg_latency": 1.5, "latencies": [1.4, 1.5, 1.6], "percentiles": {"50": 1.5, "90": 1.6, "95": 1.65, "99": 1.7}}'
            )
            (containerized_dir / "mi300x_mistral_perf_batch16_20240813_150000.json").write_text(
                '{"avg_latency": 2.0, "latencies": [1.9, 2.0, 2.1], "percentiles": {"50": 2.0, "90": 2.1, "95": 2.15, "99": 2.2}}'
            )
            (containerized_dir / "unknown_format_file.json").write_text(
                '{"avg_latency": 3.0, "latencies": [3.0], "percentiles": {"50": 3.0, "90": 3.0, "95": 3.0, "99": 3.0}}'
            )

            yield AnalysisConfig(input_dir=input_dir, output_dir=output_dir)

    def test_full_workflow(self, real_config: AnalysisConfig) -> None:
        """Test complete analyzer workflow"""
        analyzer = BenchmarkAnalyzer(real_config)

        # Test parsing different filename formats
        test_files = [
            "llama3_latency_bs32_in128_out256_float16_mem0.9_20240812_143022.json",
            "mi300x_mistral_perf_batch16_20240813_150000.json",
            "unknown_format_file.json",
        ]

        results = []
        for filename in test_files:
            result = analyzer._parse_experiment_filename(filename)
            results.append(result)

        # Verify parsing worked correctly
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
