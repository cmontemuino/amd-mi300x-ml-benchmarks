"""Integration tests for BenchmarkAnalyzer"""

import tempfile
from pathlib import Path

import pytest

from amd_bench.core.analysis import BenchmarkAnalyzer
from amd_bench.schemas.benchmark import AnalysisConfig


class TestBenchmarkAnalyzerIntegration:
    """Integration tests for BenchmarkAnalyzer with real file scenarios"""

    @pytest.fixture
    def real_config(self):
        """Real configuration for integration testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"

            input_dir.mkdir()

            # Create some test files
            (
                input_dir / "llama3_latency_bs32_in128_out256_float16_mem0.9_20240812_143022.json"
            ).touch()
            (input_dir / "mi300x_mistral_perf_batch16_20240813_150000.json").touch()
            (input_dir / "unknown_format_file.json").touch()

            yield AnalysisConfig(input_dir=input_dir, output_dir=output_dir)

    def test_full_workflow(self, real_config):
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
