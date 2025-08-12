"""Core analysis functionality for benchmark results"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

from ..schemas.benchmark import (
    AnalysisConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    ExperimentConfig,
    FilenameFormat,
)
from ..utils.logging import get_logger
from ..utils.paths import ensure_directory
from .reporters import ReportGenerator
from .statistics import StatisticalAnalyzer
from .visualizers import PlotGenerator

logger = get_logger(__name__)


class BenchmarkAnalyzer:
    """Comprehensive analyzer for vLLM benchmark results"""

    def __init__(self, config: AnalysisConfig):
        """Initialize analyzer with configuration"""
        self.config: AnalysisConfig = config
        self.results: List[BenchmarkResult] = []

        # Convert config to FilenameFormat objects for processing
        self.filename_formats = self._build_filename_formats()

        # Ensure output directories exist
        self._setup_output_directories()

        logger.info("Analyzer initialized with custom filename formats")
        logger.info(f"Loaded {len(self.filename_formats)} filename formats")
        logger.info(f"Input directory: {self.config.input_dir}")
        logger.info(f"Output directory: {self.config.output_dir}")

    def _build_filename_formats(self) -> List[FilenameFormat]:
        """Build FilenameFormat objects from configuration"""
        formats = []
        for fmt_config in self.config.filename_formats:
            formats.append(
                FilenameFormat(
                    pattern=fmt_config["pattern"],
                    groups=fmt_config["groups"],
                    description=fmt_config.get("description", "Custom format"),
                    priority=fmt_config.get("priority", 100),
                )
            )

        # Sort by priority (lower numbers first)
        formats.sort(key=lambda x: x.priority)
        return formats

    def _setup_output_directories(self) -> None:
        """Create necessary output directories"""
        dirs = ["tables", "plots", "reports"]
        all(ensure_directory(self.config.output_dir / d) for d in dirs)

    def _parse_experiment_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse experiment filename to extract configuration parameters

        Args:
            filename: The filename to parse (can be full path or basename)

        Returns:
            Dictionary of extracted parameters with defaults for missing values
        """
        # Get basename and remove extension
        file_path = (
            self.config.input_dir / filename if not Path(filename).is_absolute() else Path(filename)
        )
        name = file_path.stem

        logger.debug(f"Parsing filename: {name}")

        # Try each format pattern
        for format_config in self.filename_formats:
            match = re.match(format_config.pattern, name)
            if match:
                logger.debug(f"Matched format: {format_config.description}")
                return self._extract_parameters_from_match(match, format_config, name)

        # No patterns matched - return minimal info with warning
        logger.warning(f"No filename pattern matched: {filename}")
        logger.warning("Supported formats:")

        return self._get_default_parameters_from_config(name)

    def _extract_parameters_from_match(
        self, match: re.Match[str], format_config: FilenameFormat, filename: str
    ) -> Dict[str, str]:
        """Extract and validate parameters from regex match"""
        # Start with configured defaults
        params = self._get_default_parameters_from_config(filename)

        try:
            if match is None:
                raise ValueError("No regex match found") from None

            # Extract matched groups
            for field, group_idx in format_config.groups.items():
                if group_idx <= len(match.groups()):
                    raw_value = match.group(group_idx)
                    params[field] = self._sanitize_parameter_value_from_config(field, raw_value)

            # Validate required parameters
            self._validate_extracted_parameters(params, filename)
        except (IndexError, ValueError) as e:
            logger.error(f"Error extracting parameters from {filename}: {e}")
            # Fall back to defaults but keep what we extracted successfully already

        return params

    def _sanitize_parameter_value_from_config(self, field: str, raw_value: str) -> str:
        """Sanitize parameter value using configured rules"""
        sanitizer = self.config.parameter_sanitizers.get(field)

        if sanitizer == "decimal_separator":
            return raw_value.replace(",", ".")
        elif sanitizer == "numeric_only":
            return re.sub(r"\D", "", raw_value)
        else:
            return raw_value.strip()

    def _get_default_parameters_from_config(self, filename: str) -> Dict[str, str]:
        """Get default parameters from configuration"""
        defaults = self.config.default_parameters.copy()
        defaults["model"] = filename  # Override with filename as model fallback
        return defaults

    @staticmethod
    def _get_default_parameters(filename: str) -> Dict[str, str]:
        """Return default parameters with fallback values"""
        return {
            "model": filename,
            "benchmark_type": "unknown",
            "batch_size": "1",
            "input_length": "512",
            "output_length": "128",
            "dtype": "float16",
            "memory_util": "0.0",
            "timestamp": "unknown",
        }

    @staticmethod
    def _sanitize_parameter_value(field: str, raw_value: str) -> str:
        """Sanitize and normalize parameter values"""
        if field == "memory_util":
            # Handle different decimal separators
            return raw_value.replace(",", ".")
        elif field in ["batch_size", "input_length", "output_length"]:
            # Ensure numeric values are sanitized
            return re.sub(r"\D", "", raw_value)
        else:
            # Apply vanilla sanitization
            return raw_value.strip()

    def _validate_extracted_parameters(self, params: Dict[str, str], filename: str) -> None:
        """Validate the extracted parameters make sense"""
        # Check numeric parameters
        numeric_fields = ["batch_size", "input_length", "output_length", "memory_util"]
        for field in numeric_fields:
            if field in params and params[field]:
                try:
                    value = float(params[field])
                    if field == "memory_util" and (value <= 0 or value > 1):
                        logger.warning(f"Unusual memory utilization in {filename}: {value}")
                    elif field != "memory_util" and value <= 0:
                        logger.warning(f"Invalid {field} in {filename}: {value}")
                except ValueError:
                    logger.warning(f"Non-numeric {field} in {filename}: {params[field]}")

        # Validate timestamp format
        if "timestamp" in params and params["timestamp"]:
            self._validate_timestamp_format(params["timestamp"], filename)

    @staticmethod
    def _validate_timestamp_format(timestamp: str, filename: str) -> None:
        """Validate timestamp format and provide helpful warnings."""
        common_formats = [
            ("%Y%m%d_%H%M%S", "YYYYMMDD_HHMMSS"),
            ("%Y-%m-%d_%H-%M-%S", "YYYY-MM-DD_HH-MM-SS"),
            ("%Y%m%d%H%M%S", "YYYYMMDDHHMMSS"),
            ("%Y%m%d", "YYYYMMDD"),
        ]

        from datetime import datetime

        for fmt_string, description in common_formats:
            try:
                datetime.strptime(timestamp, fmt_string)
                logger.debug(f"Timestamp format validated: {description}")
                return
            except ValueError:
                continue

        logger.warning(f"Unrecognized timestamp format in {filename}: {timestamp}")
        logger.info("Consider using format: YYYYMMDD_HHMMSS")

    def process_results(self) -> None:
        """
        Process all benchmark result files and generate comprehensive analysis

        This method orchestrates the complete analysis workflow:
        1. Discover and parse result files
        2. Load and validate benchmark data
        3. Generate statistical summaries
        4. Create visualization outputs
        5. Export analysis reports
        """
        logger.info("Starting benchmark results processing...")

        try:
            # Step 1: Discover result files
            result_files = self._discover_result_files()
            logger.info(f"Found {len(result_files)} result files to process")

            if not result_files:
                logger.warning("No result files found matching the specified pattern")
                return

            # Step 2: Load and parse all results
            self.results = self._load_all_results(result_files)
            logger.info(f"Successfully loaded {len(self.results)} benchmark results")

            if not self.results:
                logger.error("No valid results could be loaded")
                return

            # Step 3: Generate analysis outputs
            stats_analyzer = StatisticalAnalyzer(self.results)
            stats_analyzer.export_summaries(self.config.output_dir)

            # 4. Create visualization outputs
            if self.config.generate_plots:
                plot_generator = PlotGenerator(self.results)
                plot_generator.create_all_plots(self.config.output_dir / "plots")

            # 5. Export analysis reports
            report_generator = ReportGenerator(self.config, self.results, stats_analyzer)
            report_generator.create_reports(self.config.output_dir / "reports")

            logger.info("Benchmark analysis completed successfully")
            logger.info(f"Results available in: {self.config.output_dir}")

        except Exception as e:
            logger.error(f"Error during results processing: {e}")
            raise

    def _discover_result_files(self) -> List[Path]:
        """Discover all result files matching the configured pattern"""
        logger.debug(f"Searching for files matching: {self.config.results_pattern}")
        logger.debug(f"In directory: {self.config.input_dir}")

        result_files = list(self.config.input_dir.glob(self.config.results_pattern))

        # Filter out non-JSON files if pattern is generic
        json_files = [f for f in result_files if f.suffix.lower() == ".json"]

        logger.debug(f"Found {len(json_files)} JSON files")
        return sorted(json_files)

    def _load_all_results(self, result_files: List[Path]) -> List[BenchmarkResult]:
        """Load and validate all benchmark result files"""
        results = []
        failed_files = []

        for file_path in result_files:
            try:
                result = self._load_single_result(file_path)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
                failed_files.append(file_path.name)

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files[:5]}...")

        return results

    def _load_single_result(self, file_path: Path) -> Optional[BenchmarkResult]:
        """Load and validate a single benchmark result file"""
        logger.debug(f"Loading result file: {file_path.name}")

        try:
            # Load JSON data
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Parse filename to extract configuration
            config_params = self._parse_experiment_filename(file_path.name)

            # Create experiment configuration
            experiment_config = self._create_experiment_config(config_params)

            # Extract and validate metrics
            metrics = self._extract_benchmark_metrics(raw_data)

            # Create complete benchmark result
            result = BenchmarkResult(
                file=file_path.name,
                experiment_id=self._generate_experiment_id(config_params),
                config=experiment_config,
                metrics=metrics,
            )

            logger.debug(f"Successfully loaded: {file_path.name}")
            return result

        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            return None

    @staticmethod
    def _create_experiment_config(params: Dict[str, str]) -> ExperimentConfig:
        """Create ExperimentConfig from parsed parameters"""
        return ExperimentConfig(
            model=params.get("model", "unknown"),
            benchmark_type=cast(
                Literal["latency", "throughput"], params.get("benchmark_type", "latency")
            ),
            batch_size=int(params.get("batch_size", "1")),
            input_length=int(params.get("input_length", "128")),
            output_length=int(params.get("output_length", "128")),
            dtype=cast(
                Literal["float16", "bfloat16", "float8", "int4"], params.get("dtype", "float16")
            ),
            memory_util=float(params.get("memory_util", "0.8")),
            timestamp=params.get("timestamp", "unknown"),
        )

    def _extract_benchmark_metrics(self, raw_data: Dict[str, Any]) -> BenchmarkMetrics:
        """Extract and validate benchmark metrics from raw JSON data"""
        # Handle different JSON structures
        if "avg_latency" in raw_data:
            # vLLM latency format
            return self._extract_vllm_latency_metrics(raw_data)
        elif "throughput" in raw_data:
            # Throughput format
            return self._extract_throughput_metrics(raw_data)
        else:
            raise ValueError(f"Unrecognized benchmark data format: {list(raw_data.keys())}")

    @staticmethod
    def _extract_vllm_latency_metrics(data: Dict[str, Any]) -> BenchmarkMetrics:
        """Extract metrics from vLLM latency benchmark results"""
        latencies = data.get("latencies", [])
        percentiles = data.get("percentiles", {})

        # Calculate additional statistics if not provided
        import statistics

        return BenchmarkMetrics(
            avg_latency=data.get("avg_latency", 0.0),
            latency_std=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            p50_latency=percentiles.get("50", percentiles.get(50, 0.0)),
            p90_latency=percentiles.get("90", percentiles.get(90, 0.0)),
            p95_latency=percentiles.get("95", percentiles.get(95, 0.0)),
            p99_latency=percentiles.get("99", percentiles.get(99, 0.0)),
            throughput=1.0 / data.get("avg_latency", 1.0),  # Requests per second
            tokens_per_second=0.0,  # Not available in latency data
            total_iterations=len(latencies),
        )

    @staticmethod
    def _extract_throughput_metrics(data: Dict[str, Any]) -> BenchmarkMetrics:
        """Extract metrics from throughput benchmark results"""
        return BenchmarkMetrics(
            avg_latency=data.get("avg_latency", 0.0),
            latency_std=data.get("latency_std", 0.0),
            p50_latency=data.get("p50_latency", 0.0),
            p90_latency=data.get("p90_latency", 0.0),
            p95_latency=data.get("p95_latency", 0.0),
            p99_latency=data.get("p99_latency", 0.0),
            throughput=data.get("throughput", 0.0),
            tokens_per_second=data.get("tokens_per_second", 0.0),
            total_iterations=data.get("total_iterations", 0),
        )

    @staticmethod
    def _generate_experiment_id(params: Dict[str, str]) -> str:
        """Generate unique experiment identifier"""
        key_params = [
            params.get("model", "unknown"),
            params.get("benchmark_type", "unknown"),
            params.get("batch_size", "1"),
            params.get("memory_util", "0.0"),
            params.get("timestamp", "unknown"),
        ]
        return "_".join(str(p).replace("/", "-") for p in key_params)
