"""Unit tests for analysis core functionality"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from amd_bench.core.analysis import BenchmarkAnalyzer
from amd_bench.schemas.benchmark import AnalysisConfig, FilenameFormat


class TestBenchmarkAnalyzerInstanceMethods:
    """Test instance methods of BenchmarkAnalyzer"""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            yield input_dir, output_dir

    @pytest.fixture
    def basic_config(self, temp_dirs):
        """Basic AnalysisConfig for testing"""
        input_dir, output_dir = temp_dirs
        return AnalysisConfig(input_dir=input_dir, output_dir=output_dir)

    @pytest.fixture
    def custom_config(self, temp_dirs):
        """Custom AnalysisConfig with custom formats"""
        input_dir, output_dir = temp_dirs
        return AnalysisConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            filename_formats=[
                {
                    "pattern": r"test_([^_]+)_batch(\d+)_(.+)",
                    "groups": {"model": 1, "batch_size": 2, "timestamp": 3},
                    "description": "Test format",
                    "priority": 1,
                },
                {
                    "pattern": r"([^_]+)_standard_(.+)",
                    "groups": {"model": 1, "timestamp": 2},
                    "description": "Standard test format",
                    "priority": 2,
                },
            ],
            default_parameters={"model": "test_model", "benchmark_type": "test_benchmark"},
            parameter_sanitizers={"batch_size": "numeric_only", "memory_util": "decimal_separator"},
        )

    class TestInitialization:
        """Test BenchmarkAnalyzer initialization"""

        def test_basic_initialization(self, basic_config):
            """Test basic analyzer initialization"""
            analyzer = BenchmarkAnalyzer(basic_config)

            assert analyzer.config == basic_config
            assert analyzer.results == []
            assert len(analyzer.filename_formats) == 2  # Default formats
            assert analyzer.config.output_dir.exists()

        def test_custom_initialization(self, custom_config):
            """Test analyzer initialization with custom config"""
            analyzer = BenchmarkAnalyzer(custom_config)

            assert len(analyzer.filename_formats) == 2
            assert analyzer.filename_formats[0].priority == 1
            assert analyzer.filename_formats[1].priority == 2
            assert analyzer.filename_formats[0].description == "Test format"

        @patch("amd_bench.core.analysis.logger")
        def test_initialization_logging(self, mock_logger, basic_config):
            """Test that initialization logs appropriate messages"""
            BenchmarkAnalyzer(basic_config)

            # Check that logger was called with expected messages
            mock_logger.info.assert_any_call("Analyzer initialized with custom filename formats")
            mock_logger.info.assert_any_call("Loaded 2 filename formats")

    class TestBuildFilenameFormats:
        """Test _build_filename_formats method"""

        def test_build_formats_from_config(self, custom_config):
            """Test building filename formats from configuration"""
            analyzer = BenchmarkAnalyzer(custom_config)

            formats = analyzer.filename_formats

            assert len(formats) == 2
            assert all(isinstance(fmt, FilenameFormat) for fmt in formats)

            # Check sorting by priority
            assert formats[0].priority == 1
            assert formats[1].priority == 2

            # Check content
            assert formats[0].pattern == r"test_([^_]+)_batch(\d+)_(.+)"
            assert formats[0].groups == {"model": 1, "batch_size": 2, "timestamp": 3}

        def test_build_formats_with_defaults(self, basic_config):
            """Test building formats with default configuration"""
            analyzer = BenchmarkAnalyzer(basic_config)

            formats = analyzer.filename_formats

            assert len(formats) == 2  # Default has 2 formats
            assert all(isinstance(fmt, FilenameFormat) for fmt in formats)

    class TestSetupOutputDirectories:
        """Test _setup_output_directories method"""

        @patch("amd_bench.core.analysis.ensure_directory")
        def test_setup_output_directories(self, mock_ensure_dir, basic_config):
            """Test output directory setup"""
            BenchmarkAnalyzer(basic_config)

            # Should call ensure_directory for each subdirectory
            expected_calls = [
                basic_config.output_dir / "tables",
                basic_config.output_dir / "plots",
                basic_config.output_dir / "reports",
            ]

            assert mock_ensure_dir.call_count == 3

            for expected_path in expected_calls:
                mock_ensure_dir.assert_any_call(expected_path)

    class TestParseExperimentFilename:
        """Test _parse_experiment_filename method"""

        def test_parse_filename_with_matching_pattern(self, custom_config):
            """Test parsing filename that matches a pattern"""
            analyzer = BenchmarkAnalyzer(custom_config)

            filename = "test_llama3_batch32_20240812_1430.json"
            result = analyzer._parse_experiment_filename(filename)

            assert result["model"] == "llama3"
            assert result["batch_size"] == "32"
            assert result["timestamp"] == "20240812_1430"

        def test_parse_filename_with_multiple_patterns(self, custom_config):
            """Test that higher priority patterns are matched first"""
            analyzer = BenchmarkAnalyzer(custom_config)

            # This could match both patterns, but should match the higher priority one
            filename = "test_model_batch16_timestamp.json"
            result = analyzer._parse_experiment_filename(filename)

            # Should match the first (higher priority) pattern
            assert result["model"] == "model"
            assert result["batch_size"] == "16"

        @patch("amd_bench.core.analysis.logger")
        def test_parse_filename_no_match(self, mock_logger, custom_config):
            """Test parsing filename that doesn't match any pattern"""
            analyzer = BenchmarkAnalyzer(custom_config)

            filename = "completely_different_format.json"
            result = analyzer._parse_experiment_filename(filename)

            # Should use defaults
            assert result["model"] == "completely_different_format"
            assert result["benchmark_type"] == "test_benchmark"  # From custom config

            # Should log warning
            mock_logger.warning.assert_called()

        def test_parse_absolute_path(self, custom_config):
            """Test parsing with absolute path"""
            analyzer = BenchmarkAnalyzer(custom_config)

            abs_path = "/absolute/path/test_model_batch8_timestamp.json"
            result = analyzer._parse_experiment_filename(abs_path)

            assert result["model"] == "model"
            assert result["batch_size"] == "8"

    class TestExtractParametersFromMatch:
        """Test _extract_parameters_from_match method"""

        def test_extract_parameters_success(self, custom_config):
            """Test successful parameter extraction"""
            import re

            analyzer = BenchmarkAnalyzer(custom_config)
            format_config = analyzer.filename_formats[0]

            match = re.match(format_config.pattern, "test_llama3_batch32_20240812")

            result = analyzer._extract_parameters_from_match(
                match, format_config, "test_llama3_batch32_20240812"
            )

            assert result["model"] == "llama3"
            assert result["batch_size"] == "32"  # Should be sanitized (numeric_only)
            assert result["timestamp"] == "20240812"

        def test_extract_parameters_with_sanitization(self, custom_config):
            """Test parameter extraction with sanitization"""
            import re

            analyzer = BenchmarkAnalyzer(custom_config)
            format_config = analyzer.filename_formats[0]

            # FIXED: Use input that actually matches the pattern
            match = re.match(format_config.pattern, "test_model_batch32_timestamp")

            # Verify match exists before proceeding
            assert match is not None, "Pattern should match the test input"

            result = analyzer._extract_parameters_from_match(match, format_config, "test_filename")

            # batch_size should be extracted as "32" (already clean digits)
            assert result["batch_size"] == "32"
            assert result["model"] == "model"
            assert result["timestamp"] == "timestamp"

        def test_extract_parameters_with_sanitization_complex(self, custom_config):
            """Test parameter extraction with complex sanitization needs"""
            import re

            analyzer = BenchmarkAnalyzer(custom_config)

            # Create a custom format that allows more flexible batch_size matching
            complex_format = FilenameFormat(
                pattern=r"test_([^_]+)_batch([^_]+)_(.+)",
                groups={"model": 1, "batch_size": 2, "timestamp": 3},
                description="Complex test format",
                priority=1,
            )

            match = re.match(complex_format.pattern, "test_model_batch32tokens_timestamp")
            assert match is not None

            result = analyzer._extract_parameters_from_match(match, complex_format, "test_filename")

            # batch_size should have "tokens" removed due to numeric_only sanitizer
            assert result["batch_size"] == "32"

        @patch("amd_bench.core.analysis.logger")
        def test_extract_parameters_with_error(self, mock_logger, custom_config):
            """Test parameter extraction with errors"""
            analyzer = BenchmarkAnalyzer(custom_config)
            format_config = analyzer.filename_formats[0]

            # Create a mock match that will raise an error
            mock_match = Mock()
            mock_match.groups.return_value = ("group1", "group2", "group3")  # Return some groups

            # Make group() method raise IndexError when called with valid indices
            def side_effect(group_idx):
                if group_idx <= 3:  # This should match the groups in format_config
                    raise IndexError(f"Invalid group index: {group_idx}")
                return "default"

            mock_match.group.side_effect = side_effect

            result = analyzer._extract_parameters_from_match(
                mock_match, format_config, "test_filename"
            )

            # Should still return defaults even with error
            assert "model" in result
            # Verify error was logged
            mock_logger.error.assert_called()

        @patch("amd_bench.core.analysis.logger")
        def test_extract_parameters_with_none_match(self, mock_logger, custom_config):
            """Test parameter extraction when match is None"""
            analyzer = BenchmarkAnalyzer(custom_config)
            format_config = analyzer.filename_formats[0]

            # This should handle None match gracefully
            result = analyzer._extract_parameters_from_match(None, format_config, "test_filename")

            # Should return defaults without crashing
            assert "model" in result
            assert result["model"] == "test_filename"
            mock_logger.error.assert_called()

    class TestSanitizeParameterValueFromConfig:
        """Test _sanitize_parameter_value_from_config method"""

        def test_decimal_separator_sanitization(self, custom_config):
            """Test decimal separator sanitization"""
            analyzer = BenchmarkAnalyzer(custom_config)

            result = analyzer._sanitize_parameter_value_from_config("memory_util", "0,95")
            assert result == "0.95"

        def test_numeric_only_sanitization(self, custom_config):
            """Test numeric only sanitization"""
            analyzer = BenchmarkAnalyzer(custom_config)

            result = analyzer._sanitize_parameter_value_from_config("batch_size", "32tokens")
            assert result == "32"

        def test_no_sanitization(self, custom_config):
            """Test fields without specific sanitization"""
            analyzer = BenchmarkAnalyzer(custom_config)

            result = analyzer._sanitize_parameter_value_from_config("model", "  llama3  ")
            assert result == "llama3"

    class TestGetDefaultParametersFromConfig:
        """Test _get_default_parameters_from_config method"""

        def test_get_defaults_with_custom_config(self, custom_config):
            """Test getting defaults from custom configuration"""
            analyzer = BenchmarkAnalyzer(custom_config)

            result = analyzer._get_default_parameters_from_config("test_filename")

            # Should use custom defaults
            assert result["benchmark_type"] == "test_benchmark"
            assert result["model"] == "test_filename"  # Should override with filename

        def test_get_defaults_preserves_config(self, custom_config):
            """Test that getting defaults doesn't modify original config"""
            analyzer = BenchmarkAnalyzer(custom_config)

            original_defaults = analyzer.config.default_parameters.copy()

            analyzer._get_default_parameters_from_config("test_filename")

            # Original config should be unchanged
            assert analyzer.config.default_parameters == original_defaults

    class TestValidateExtractedParameters:
        """Test _validate_extracted_parameters method"""

        @patch("amd_bench.core.analysis.logger")
        def test_validate_parameters_success(self, mock_logger, basic_config):
            """Test successful parameter validation"""
            analyzer = BenchmarkAnalyzer(basic_config)

            params = {
                "batch_size": "32",
                "input_length": "128",
                "output_length": "256",
                "memory_util": "0.9",
                "timestamp": "20240812_143022",
            }

            # Should not raise exceptions
            analyzer._validate_extracted_parameters(params, "test_file.json")

            # Should not log warnings for valid parameters
            warning_calls = list(mock_logger.warning.call_args_list)
            assert len(warning_calls) == 0

        @patch("amd_bench.core.analysis.logger")
        def test_validate_parameters_with_warnings(self, mock_logger, basic_config):
            """Test parameter validation with warnings"""
            analyzer = BenchmarkAnalyzer(basic_config)

            params = {
                "batch_size": "invalid",
                "memory_util": "1.5",  # > 1
                "timestamp": "invalid_timestamp",
            }

            analyzer._validate_extracted_parameters(params, "test_file.json")

            # Should log warnings for invalid parameters
            assert mock_logger.warning.call_count >= 2

        @patch("amd_bench.core.analysis.logger")
        def test_validate_empty_parameters(self, mock_logger, basic_config):
            """Test validation with empty parameters"""
            analyzer = BenchmarkAnalyzer(basic_config)

            # Should not crash with empty parameters
            analyzer._validate_extracted_parameters({}, "test_file.json")


class TestBenchmarkAnalyzerStaticMethods:
    """Test static utility methods from BenchmarkAnalyzer"""

    class TestGetDefaultParameters:
        """Test _get_default_parameters static method"""

        def test_basic_filename(self):
            """Test with basic filename"""
            result = BenchmarkAnalyzer._get_default_parameters("test_file")

            expected = {
                "model": "test_file",
                "benchmark_type": "unknown",
                "batch_size": "1",
                "input_length": "512",
                "output_length": "128",
                "dtype": "float16",
                "memory_util": "0.0",
                "timestamp": "unknown",
            }

            assert result == expected

        def test_complex_filename(self):
            """Test with complex filename containing underscores"""
            result = BenchmarkAnalyzer._get_default_parameters("llama3-8b_complex_name")

            assert result["model"] == "llama3-8b_complex_name"
            assert result["benchmark_type"] == "unknown"

        def test_empty_filename(self):
            """Test with empty filename"""
            result = BenchmarkAnalyzer._get_default_parameters("")

            assert result["model"] == ""
            assert result["benchmark_type"] == "unknown"

        def test_returned_keys_completeness(self):
            """Test that all required keys are present"""
            result = BenchmarkAnalyzer._get_default_parameters("tests")

            required_keys = {
                "model",
                "benchmark_type",
                "batch_size",
                "input_length",
                "output_length",
                "dtype",
                "memory_util",
                "timestamp",
            }

            assert set(result.keys()) == required_keys

    class TestSanitizeParameterValue:
        """Test _sanitize_parameter_value static method"""

        def test_memory_util_comma_replacement(self):
            """Test memory_util comma to dot replacement"""
            result = BenchmarkAnalyzer._sanitize_parameter_value("memory_util", "0,95")
            assert result == "0.95"

        @pytest.mark.parametrize("field", ["batch_size", "input_length", "output_length"])
        def test_numeric_fields_sanitization(self, field):
            """Test numeric fields remove non-digits"""
            result = BenchmarkAnalyzer._sanitize_parameter_value(field, "abc123def456")
            assert result == "123456"

        def test_other_fields_strip_whitespace(self):
            """Test other fields just strip whitespace"""
            result = BenchmarkAnalyzer._sanitize_parameter_value("model", "  llama-7b  ")
            assert result == "llama-7b"

    class TestValidateTimestampFormat:
        """Test _validate_timestamp_format static method"""

        @patch("amd_bench.core.analysis.logger")
        def test_valid_yyyymmdd_hhmmss_format(self, mock_logger):
            """Test valid YYYYMMDD_HHMMSS format"""
            BenchmarkAnalyzer._validate_timestamp_format("20240812_143022", "test_file.json")

            # Should not log any warnings
            mock_logger.warning.assert_not_called()

        @patch("amd_bench.core.analysis.logger")
        def test_invalid_timestamp_logs_warning(self, mock_logger):
            """Test invalid timestamp logs warning"""
            BenchmarkAnalyzer._validate_timestamp_format("invalid_timestamp", "test_file.json")

            # Should log warning
            mock_logger.warning.assert_called()

            # Verify warning content
            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) >= 1

            warning_msg = str(warning_calls[0][0][0])
            assert "Unrecognized timestamp format" in warning_msg


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_analyzer_with_invalid_regex_pattern(self):
        """Test analyzer with invalid regex pattern"""
        with pytest.raises(ValueError, match="invalid regex pattern"):
            AnalysisConfig(
                input_dir=Path("/tmp"),
                output_dir=Path("/tmp/output"),
                filename_formats=[
                    {
                        "pattern": r"[invalid(regex",  # Invalid regex
                        "groups": {"model": 1},
                        "description": "Invalid pattern",
                    }
                ],
            )

    def test_empty_parameters_handling(self):
        """Test handling of completely empty parameters"""
        config = AnalysisConfig(input_dir=Path("/tmp"), output_dir=Path("/tmp/output"))

        analyzer = BenchmarkAnalyzer(config)
        result = analyzer._get_default_parameters_from_config("")

        assert result["model"] == ""
        assert "benchmark_type" in result
