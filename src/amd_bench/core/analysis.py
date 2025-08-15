"""Core analysis functionality for benchmark results"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

import pandas as pd

from amd_bench.schemas.benchmark import (
    AnalysisConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    ExperimentConfig,
    ExperimentFiles,
    FilenameFormat,
)
from amd_bench.utils.logging import get_logger
from amd_bench.utils.paths import ensure_directory

from .reporters import ReportGenerator
from .statistics import StatisticalAnalyzer
from .visualizers import BenchmarkVisualizer

logger = get_logger(__name__)


class BenchmarkAnalyzer:
    """Comprehensive analyzer for vLLM benchmark results"""

    def __init__(self, config: AnalysisConfig):
        """Initialize analyzer with configuration"""
        self.config: AnalysisConfig = config
        self.results: List[BenchmarkResult] = []
        self.experiment_files: List[ExperimentFiles] = []

        # Build dynamic directory paths
        self.results_dir = self._get_results_directory()
        self.logs_dir = self._get_logs_directory()
        self.monitoring_dir = self._get_monitoring_directory()

        # Convert config to FilenameFormat objects for processing
        self.filename_formats = self._build_filename_formats()

        # Ensure output directories and log directory exist
        self._setup_output_directories()
        self._validate_and_log_structure()

        # Cache for tensor parallel size (parsed once per benchmark run)
        self._tensor_parallel_cache: Optional[int] = None
        self._benchmark_run_log_path: Optional[Path] = None

        logger.info("Analyzer initialized with custom filename formats")
        logger.info(f"Loaded {len(self.filename_formats)} filename formats")
        logger.info(f"Input directory: {self.config.input_dir}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Logs directory: {self.logs_dir}")
        logger.info(f"Monitoring directory: {self.monitoring_dir}")
        logger.info(f"Output directory: {self.config.output_dir}")

    def _get_tensor_parallel_size(self, benchmark_run_log: Optional[Path]) -> Optional[int]:
        """Get tensor parallel size with caching optimization."""
        # If  already parsed this log file, return cached value
        if (
            self._benchmark_run_log_path == benchmark_run_log
            and self._tensor_parallel_cache is not None
        ):
            return self._tensor_parallel_cache

        # Parse tensor parallel size from log (only once per benchmark run)
        if benchmark_run_log and benchmark_run_log.exists():
            tensor_parallel_size = self._extract_tensor_parallel_from_log(benchmark_run_log)

            # Cache the result
            self._tensor_parallel_cache = tensor_parallel_size
            self._benchmark_run_log_path = benchmark_run_log

            logger.info(
                f"Cached tensor_parallel_size={tensor_parallel_size} from {benchmark_run_log.name}"
            )
            return tensor_parallel_size

        return None

    def _get_results_directory(self) -> Path:
        """Get the directory containing JSON result files"""
        if self.config.results_subdir:
            subdir = self.config.input_dir / self.config.results_subdir
            if subdir.exists():
                return subdir
        return self.config.input_dir

    def _get_logs_directory(self) -> Optional[Path]:
        """Get the directory containing log files"""
        if self.config.logs_subdir:
            return self.config.input_dir / self.config.logs_subdir
        # Check if logs are in root
        log_files = list(self.config.input_dir.glob("*.log"))
        return self.config.input_dir if log_files else None

    def _get_monitoring_directory(self) -> Optional[Path]:
        """Get the directory containing monitoring CSV files"""
        if self.config.monitoring_subdir:
            return self.config.input_dir / self.config.monitoring_subdir
        # Check if monitoring files are in root
        csv_files = list(self.config.input_dir.glob("cpu_*.csv"))
        csv_files.extend(list(self.config.input_dir.glob("gpu_*.csv")))
        return self.config.input_dir if csv_files else None

    def _validate_and_log_structure(self) -> None:
        """Validate and log the experiment directory structure."""
        structure_info = {}

        # Check results directory
        if self.results_dir.exists():
            json_files = list(self.results_dir.glob(self.config.results_pattern))
            structure_info["results"] = len(json_files)
            logger.info(f"Found {len(json_files)} result files in {self.results_dir}")
        else:
            logger.error(f"Results directory not found: {self.results_dir}")

        # Check logs directory
        if self.logs_dir and self.logs_dir.exists():
            log_files = list(self.logs_dir.glob("*.log"))
            structure_info["logs"] = len(log_files)
            logger.info(f"Found {len(log_files)} log files in {self.logs_dir}")
        else:
            logger.debug(f"Logs directory not found: {self.logs_dir}")

        # Check monitoring directory
        if self.monitoring_dir and self.monitoring_dir.exists():
            csv_files = list(self.monitoring_dir.glob("*.csv"))
            structure_info["monitoring"] = len(csv_files)
            logger.info(f"Found {len(csv_files)} monitoring files in {self.monitoring_dir}")
        else:
            logger.debug(f"Monitoring directory not found: {self.monitoring_dir}")

    def discover_experiment_files(self) -> List[ExperimentFiles]:
        """Discover and match all files for each experiment.

        This method systematically searches for benchmark result files and their associated
        monitoring data (logs, CPU metrics, GPU power/temperature data), creating a
        comprehensive mapping of all experiment files. It handles flexible directory
        structures and provides filtering options based on completeness requirements.

        The discovery process follows these steps:
        1. Try to discover the master benchmark log run
        2. Locate all JSON result files matching the configured pattern
        3. For each result file, search for corresponding monitoring files
        4. Build ExperimentFiles objects containing all related file paths
        5. Apply filtering based on completeness requirements if specified

        Returns:
            List[ExperimentFiles]: A list of ExperimentFiles objects, each containing:
                - result_file: Path to the JSON benchmark results file
                - log_file: Optional path to execution log file
                - cpu_metrics_file: Optional path to CPU monitoring CSV
                - gpu_power_file: Optional path to GPU power monitoring CSV
                - gpu_temp_file: Optional path to GPU temperature monitoring CSV

        Raises:
            FileNotFoundError: If no result files are found matching the pattern.
            PermissionError: If files exist but are not readable.

        Note:
            The method respects the `require_complete_monitoring` configuration flag.
            When enabled, only experiments with all monitoring files are returned.
            File matching is based on basename pattern matching across subdirectories.

        Example:
            ```
            analyzer = BenchmarkAnalyzer(config)
            experiments = analyzer.discover_experiment_files()
            print(f"Found {len(experiments)} complete experiment file sets")

            for exp in experiments:
                if exp.has_complete_monitoring:
                    print(f"Complete monitoring data for {exp.result_file.name}")
            ```
        """
        logger.info("Discovering experiment files...")

        # First, find the master benchmark run log
        benchmark_run_log = self._find_benchmark_run_log()

        # Get all JSON result files
        result_files = list(self.results_dir.glob(self.config.results_pattern))
        experiments = []

        for result_file in result_files:
            try:
                experiment = self._build_experiment_files(result_file, benchmark_run_log)
                experiments.append(experiment)

                # Log what we found for this experiment
                monitoring_status = "complete" if experiment.has_complete_monitoring else "partial"
                logger.debug(f"Experiment {result_file.name}: {monitoring_status} monitoring data")

            except Exception as e:
                logger.error(f"Error processing experiment {result_file.name}: {e}")

        # Filter experiments if complete monitoring is required
        if self.config.require_complete_monitoring:
            complete_experiments = [exp for exp in experiments if exp.has_complete_monitoring]
            logger.info(
                f"Filtered to {len(complete_experiments)}/{len(experiments)} experiments with complete monitoring"
            )
            experiments = complete_experiments

        logger.info(f"Discovered {len(experiments)} experiment file sets")
        return experiments

    def _find_benchmark_run_log(self) -> Optional[Path]:
        """Find the master benchmark run log file."""
        if not self.logs_dir or not self.logs_dir.exists():
            return None

        # Look for files matching pattern: benchmark_run_*.log
        benchmark_logs = list(self.logs_dir.glob("benchmark_run_*.log"))

        if benchmark_logs:
            # If multiple logs exist, take the most recent one
            latest_log = max(benchmark_logs, key=lambda x: x.stat().st_mtime)
            logger.info(f"Found master benchmark log: {latest_log.name}")
            return latest_log

        logger.warning("No master benchmark run log found")
        return None

    def _build_experiment_files(
        self, result_file: Path, benchmark_run_log: Optional[Path]
    ) -> ExperimentFiles:
        """Build ExperimentFiles object by finding matching files."""
        # Extract the base pattern from the result filename
        base_name = result_file.stem  # Remove .json extension

        # Look for matching files in other directories
        log_file = None
        if self.logs_dir and self.logs_dir.exists():
            potential_log = self.logs_dir / f"{base_name}.log"
            if potential_log.exists():
                log_file = potential_log

        cpu_metrics_file = None
        gpu_power_file = None
        gpu_temp_file = None

        if self.monitoring_dir and self.monitoring_dir.exists():
            # Look for CPU metrics
            cpu_pattern = self.monitoring_dir / f"cpu_{base_name}.csv"
            if cpu_pattern.exists():
                cpu_metrics_file = cpu_pattern

            # Look for GPU power metrics
            power_pattern = self.monitoring_dir / f"gpu_power_{base_name}.csv"
            if power_pattern.exists():
                gpu_power_file = power_pattern

            # Look for GPU temperature metrics
            temp_pattern = self.monitoring_dir / f"gpu_temp_{base_name}.csv"
            if temp_pattern.exists():
                gpu_temp_file = temp_pattern

        return ExperimentFiles(
            result_file=result_file,
            benchmark_run_log=benchmark_run_log,  # Shared across all experiments
            log_file=log_file,
            cpu_metrics_file=cpu_metrics_file,
            gpu_power_file=gpu_power_file,
            gpu_temp_file=gpu_temp_file,
        )

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
        # Skip validation for 'unknown' timestamps
        if timestamp == "unknown":
            return

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
                return  # Successfully validated, no warning needed
            except ValueError:
                continue

        # Only log warning if no format matched
        logger.warning(f"Unrecognized timestamp format in {filename}: {timestamp}")
        logger.info("Consider using format: YYYYMMDD_HHMMSS")

    def process_results(self) -> None:
        """Process all benchmark result files and generate comprehensive analysis.

        This is the main orchestration method that coordinates the complete analysis
        workflow from raw benchmark files to final reports and visualizations. It
        handles file discovery, data loading, statistical analysis, visualization
        generation, and report creation in a fault-tolerant manner.

        The processing pipeline includes:
        1. **File Discovery**: Locate all experiment files and validate structure
        2. **Data Loading**: Parse JSON results and extract benchmark metrics
        3. **Statistical Analysis**: Generate performance summaries and comparisons
        4. **Monitoring Processing**: Analyze hardware metrics if available
        5. **Visualization**: Create performance plots and dashboards
        6. **Report Generation**: Produce markdown and JSON analysis reports

        The method implements comprehensive error handling and logging to ensure
        partial results are preserved even if individual steps fail.

        Raises:
            ValueError: If no valid experiment files are found.
            RuntimeError: If critical analysis steps fail unexpectedly.
            PermissionError: If output directories cannot be created or accessed.

        Side Effects:
            - Creates output directory structure (tables/, plots/, reports/)
            - Writes CSV files with statistical summaries
            - Generates PNG visualization files
            - Creates comprehensive analysis reports
            - Logs detailed progress and error information

        Example:
            ```
            config = AnalysisConfig(
                input_dir=Path("benchmark_data"),
                output_dir=Path("analysis_output"),
                generate_plots=True,
                include_monitoring_data=True
            )

            analyzer = BenchmarkAnalyzer(config)
            analyzer.process_results()

            # Results available in:
            # - analysis_output/tables/*.csv
            # - analysis_output/plots/*.png
            # - analysis_output/reports/*.{md,json}
            ```

        Note:
            Processing time scales with dataset size and enabled features.
            Large datasets with monitoring data may require several minutes.
            Progress is logged at INFO level for monitoring long-running analyses.
        """
        logger.info("Starting benchmark results processing...")

        try:
            # Discover all experiment files
            self.experiment_files = self.discover_experiment_files()

            if not self.experiment_files:
                logger.warning("No valid experiment files found")
                return

            # Load benchmark results
            self.results = self._load_benchmark_results()

            if not self.results:
                logger.error("No valid results could be loaded")
                return

            # Generate statistical analysis
            stats_analyzer = StatisticalAnalyzer(self.results)
            stats_analyzer.export_summaries(self.config.output_dir)

            # Process monitoring data
            monitoring_dataframes = self._process_monitoring_data()

            # Generate visualizations
            if self.config.generate_plots:
                plot_generator = BenchmarkVisualizer(
                    self.results,
                    config=AnalysisConfig(
                        input_dir=self.config.input_dir, output_dir=Path("plots")
                    ),
                )
                standard_plots = plot_generator.create_all_plots(self.config.output_dir / "plots")

                # Create monitoring plots with in-memory data
                if monitoring_dataframes:
                    monitoring_plots = self._create_monitoring_visualizations(monitoring_dataframes)
                    standard_plots.update(monitoring_plots)

            # Generate reports
            report_generator = ReportGenerator(self.config, self.results, stats_analyzer)
            report_generator.create_reports(
                self.config.output_dir / "reports", monitoring_dataframes
            )

            logger.info("Benchmark analysis completed successfully")
            logger.info(f"Results available in: {self.config.output_dir}")

        except Exception as e:
            logger.error(f"Error during analysis processing: {e}")
            raise

    def _create_monitoring_visualizations(
        self, monitoring_dataframes: Dict[str, pd.DataFrame]
    ) -> Dict[str, Path]:
        """Create monitoring-specific visualizations."""
        monitoring_plots = {}

        try:
            # Extract monitoring data directly from the passed DataFrames
            monitoring_summaries = monitoring_dataframes.get("monitoring_summary", pd.DataFrame())
            thermal_analysis = monitoring_dataframes.get("thermal_analysis", pd.DataFrame())
            power_analysis = monitoring_dataframes.get("power_analysis", pd.DataFrame())

            # Create visualizations using the loaded data
            if (
                not monitoring_summaries.empty
                or not thermal_analysis.empty
                or not power_analysis.empty
            ):
                visualizer = BenchmarkVisualizer(self.results, self.config)

                # Create monitoring dashboard
                dashboard_plots = visualizer.create_monitoring_dashboard(
                    self.config.output_dir / "plots",
                    monitoring_summaries,
                    thermal_analysis,
                    power_analysis,
                )
                monitoring_plots.update(dashboard_plots)

                # Create power efficiency plots
                if not power_analysis.empty:
                    power_plots = visualizer.create_power_efficiency_plots(
                        self.config.output_dir / "plots", power_analysis
                    )
                    monitoring_plots.update(power_plots)

            logger.info(f"Created {len(monitoring_plots)} monitoring visualization plots")

        except Exception as e:
            logger.error(f"Error creating monitoring visualizations: {e}")

        return monitoring_plots

    def _load_benchmark_results(self) -> List[BenchmarkResult]:
        """Load benchmark results from discovered experiments."""
        results = []

        for experiment in self.experiment_files:
            try:
                # Use existing loading logic
                result = self._load_single_result(experiment.result_file)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed to load result {experiment.result_file.name}: {e}")

        return results

    def _process_monitoring_data(self) -> Dict[str, pd.DataFrame]:
        """Enhanced monitoring data processing with comprehensive metrics."""
        if not self.config.include_monitoring_data:
            return {}

        logger.info("Processing comprehensive monitoring data...")

        monitoring_summaries = []
        thermal_analysis = []
        power_analysis = []

        # Get tensor parallel size once for the entire benchmark run
        benchmark_run_log = None
        if self.experiment_files:
            benchmark_run_log = self.experiment_files[0].benchmark_run_log

        tensor_parallel_size = self._get_tensor_parallel_size(benchmark_run_log)

        for experiment in self.experiment_files:
            try:
                monitoring_data = self.load_monitoring_data(experiment)

                if monitoring_data:
                    # Core monitoring summary
                    summary = self._calculate_monitoring_summary(experiment, monitoring_data)
                    monitoring_summaries.append(summary)

                    # Enhanced thermal analysis
                    if "gpu_temp" in monitoring_data:
                        thermal_data = self._analyze_thermal_performance(
                            experiment, monitoring_data["gpu_temp"]
                        )
                        thermal_analysis.append(thermal_data)

                    # Enhanced power analysis
                    if "gpu_power" in monitoring_data:
                        power_data = self._analyze_power_efficiency(
                            experiment=experiment,
                            power_df=monitoring_data["gpu_power"],
                            tensor_parallel_size=tensor_parallel_size,
                        )
                        power_analysis.append(power_data)

            except Exception as e:
                logger.error(f"Error processing monitoring for {experiment.result_file.name}: {e}")

        gpu_allocation_summary = self._analyze_gpu_device_allocation(power_analysis)

        # Export comprehensive monitoring analysis
        return self._export_monitoring_analysis(
            monitoring_summaries, thermal_analysis, power_analysis, gpu_allocation_summary
        )

    @staticmethod
    def _analyze_gpu_device_allocation(power_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze GPU device allocation patterns across experiments."""
        if not power_analysis:
            return {}

        # Get unique devices across all experiments
        all_allocated_devices = []
        device_usage_count: Dict[str, int] = {}

        for experiment_data in power_analysis:
            if (
                "allocated_gpu_devices" in experiment_data
                and experiment_data["allocated_gpu_devices"]
            ):
                devices = experiment_data["allocated_gpu_devices"]
                all_allocated_devices.extend(devices)

                # Count usage per device
                for device in devices:
                    device_usage_count[device] = device_usage_count.get(device, 0) + 1

        unique_devices = list(set(all_allocated_devices))

        allocation_summary = {
            "unique_allocated_devices": unique_devices,
            "total_unique_devices": len(unique_devices),
            "device_usage_frequency": device_usage_count,
            "most_used_device": (
                max(device_usage_count.items(), key=lambda x: x[1]) if device_usage_count else None
            ),
        }

        logger.info(f"GPU allocation analysis: {allocation_summary}")
        return allocation_summary

    @staticmethod
    def _extract_tensor_parallel_from_log(benchmark_run_log: Optional[Path]) -> Optional[int]:
        """Extract tensor parallel size from master benchmark run log for specific experiment."""
        if not benchmark_run_log or not benchmark_run_log.exists():
            return None

        try:
            with open(benchmark_run_log, "r", encoding="utf-8") as f:
                # Just find the first occurrence of "Tensor Parallel:"
                # since it's consistent across all experiments
                for line in f:
                    if "Tensor Parallel:" in line:
                        match = re.search(r"Tensor Parallel:\s*(\d+)", line)
                        if match:
                            tensor_parallel = int(match.group(1))
                            logger.debug(
                                f"Found tensor_parallel_size={tensor_parallel} (consistent across all experiments)"
                            )
                            return tensor_parallel

            logger.warning("No 'Tensor Parallel:' line found in benchmark log")
            return None

        except Exception as e:
            logger.warning(f"Failed to parse tensor parallel size from {benchmark_run_log}: {e}")
            return None

    def _export_monitoring_analysis(
        self,
        monitoring_summaries: List[Dict[str, Any]],
        thermal_analysis: List[Dict[str, Any]],
        power_analysis: List[Dict[str, Any]],
        gpu_allocation_summary: Dict[str, Any],
    ) -> Dict[str, pd.DataFrame]:
        """Export comprehensive monitoring analysis to files."""

        monitoring_df = pd.DataFrame(monitoring_summaries)
        thermal_df = pd.DataFrame(thermal_analysis)
        power_df = pd.DataFrame(power_analysis)

        # Export general monitoring summaries
        if monitoring_summaries:
            monitoring_file = self.config.output_dir / "tables" / "monitoring_summary.csv"
            monitoring_df.to_csv(monitoring_file, index=False)
            logger.info(f"Monitoring summary exported to {monitoring_file}")

        # Export thermal analysis
        if thermal_analysis:
            thermal_file = self.config.output_dir / "tables" / "thermal_analysis.csv"
            thermal_df.to_csv(thermal_file, index=False)
            logger.info(f"Thermal analysis exported to {thermal_file}")

        # Export power analysis
        if power_analysis:
            power_file = self.config.output_dir / "tables" / "power_analysis.csv"
            power_df.to_csv(power_file, index=False)
            logger.info(f"Power analysis exported to {power_file}")

        # Export GPU allocation summary
        if gpu_allocation_summary:
            allocation_file = self.config.output_dir / "tables" / "gpu_allocation_summary.csv"
            # Convert summary to DataFrame for consistent export
            allocation_df = pd.DataFrame([gpu_allocation_summary])
            allocation_df.to_csv(allocation_file, index=False)
            logger.info(f"GPU allocation summary exported to {allocation_file}")

        return {
            "monitoring_summary": monitoring_df,  # Aggregated monitoring metrics
            "thermal_analysis": thermal_df,  # Temperature analysis data
            "power_analysis": power_df,  # Power consumption analysis
            "gpu_allocation_summary": (
                pd.DataFrame([gpu_allocation_summary]) if gpu_allocation_summary else pd.DataFrame()
            ),
        }

    @staticmethod
    def _analyze_thermal_performance(
        experiment: ExperimentFiles, temp_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze thermal performance patterns."""
        return {
            "experiment": experiment.result_file.name,
            "max_edge_temp": temp_df["temp_edge_celsius"].max(),
            "avg_edge_temp": temp_df["temp_edge_celsius"].mean(),
            "max_junction_temp": temp_df["temp_junction_celsius"].max(),
            "avg_junction_temp": temp_df["temp_junction_celsius"].mean(),
            "thermal_throttling_risk": temp_df["temp_junction_celsius"].max()
            > 90,  # AMD MI300X threshold
            "temp_stability": temp_df["temp_edge_celsius"].std(),
        }

    @staticmethod
    def _analyze_power_efficiency(
        experiment: ExperimentFiles,
        power_df: pd.DataFrame,
        tensor_parallel_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze power consumption efficiency."""
        total_power_series = power_df.groupby("timestamp")["power_watts"].sum()
        num_gpus_monitored = power_df["device"].nunique()

        results = {
            "experiment": experiment.result_file.name,
            "avg_total_power": total_power_series.mean(),
            "max_total_power": total_power_series.max(),
            "power_efficiency_all": (
                total_power_series.mean() / num_gpus_monitored if num_gpus_monitored > 0 else 0.0
            ),  # Per GPU average
            "power_stability": total_power_series.std(),
            "num_gpus_monitored": num_gpus_monitored,
            "tensor_parallel_size": tensor_parallel_size or 1,  # Store for reference
        }

        # If we know tensor parallel size, calculate allocated GPU metrics
        if tensor_parallel_size and tensor_parallel_size > 0:
            # Approach: Identify most active GPUs based on power consumption
            gpu_avg_power = (
                power_df.groupby("device")["power_watts"].mean().sort_values(ascending=False)
            )

            if len(gpu_avg_power) >= tensor_parallel_size:
                # Take the top N GPUs by power consumption (most likely the allocated ones)
                allocated_gpus = gpu_avg_power.head(tensor_parallel_size)
                allocated_devices = allocated_gpus.index.tolist()

                # Calculate power metrics for allocated GPUs only
                allocated_power_df = power_df[power_df["device"].isin(allocated_devices)]
                allocated_power_series = allocated_power_df.groupby("timestamp")[
                    "power_watts"
                ].sum()

                results.update(
                    {
                        "avg_allocated_power": allocated_power_series.mean(),
                        "power_efficiency_allocated": allocated_power_series.mean()
                        / tensor_parallel_size,
                        "num_active_gpus": tensor_parallel_size,
                        "allocated_gpu_devices": allocated_devices,
                    }
                )
            else:
                # Fallback: use all available GPUs
                results.update(
                    {
                        "avg_allocated_power": total_power_series.mean(),
                        "power_efficiency_allocated": total_power_series.mean()
                        / len(gpu_avg_power),
                        "num_active_gpus": len(gpu_avg_power),
                        "allocated_gpu_devices": gpu_avg_power.index.tolist(),
                    }
                )

        return results

    @staticmethod
    def load_monitoring_data(experiment: ExperimentFiles) -> Dict[str, pd.DataFrame]:
        """Load monitoring data for an experiment from CSV files.

        This method loads and preprocesses hardware monitoring data associated with
        a specific experiment, including CPU utilization, GPU power consumption,
        and thermal metrics. It handles multiple file formats and performs data
        validation and timestamp normalization.

        The method processes three types of monitoring data:
        - **CPU Metrics**: System utilization, load averages, idle percentages
        - **GPU Power**: Per-device power consumption over time
        - **GPU Temperature**: Edge and junction temperatures for thermal analysis

        Args:
            experiment (ExperimentFiles): Container with paths to monitoring files.
                Must contain at least one of: cpu_metrics_file, gpu_power_file,
                gpu_temp_file. Missing files are silently skipped.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping data types to DataFrames:
                - 'cpu': CPU monitoring data with timestamp column
                - 'gpu_power': GPU power consumption data
                - 'gpu_temp': GPU temperature monitoring data

            Each DataFrame includes a standardized 'timestamp' column converted
            to pandas datetime format for time-series analysis.

        Raises:
            FileNotFoundError: If specified monitoring files don't exist.
            pd.errors.EmptyDataError: If CSV files are empty or malformed.
            ValueError: If timestamp columns cannot be parsed.

        Example:
            ```
            experiment = ExperimentFiles(
                result_file=Path("result.json"),
                cpu_metrics_file=Path("cpu_metrics.csv"),
                gpu_power_file=Path("gpu_power.csv")
            )

            monitoring_data = BenchmarkAnalyzer.load_monitoring_data(experiment)

            if 'cpu' in monitoring_data:
                cpu_df = monitoring_data['cpu']
                print(f"CPU monitoring duration: {cpu_df['timestamp'].max() - cpu_df['timestamp'].min()}")

            if 'gpu_power' in monitoring_data:
                power_df = monitoring_data['gpu_power']
                total_power = power_df.groupby('timestamp')['power_watts'].sum()
                print(f"Average total power: {total_power.mean():.1f}W")
            ```

        Note:
            - Timestamps are expected in Unix epoch format (seconds since 1970)
            - GPU data may contain multiple devices with separate readings
            - Missing or corrupted files are logged as errors but don't raise exceptions
            - Empty DataFrames are returned for missing monitoring categories
        """
        monitoring_data = {}

        try:
            # Load CPU metrics
            if experiment.cpu_metrics_file and experiment.cpu_metrics_file.exists():
                cpu_df = pd.read_csv(experiment.cpu_metrics_file)
                cpu_df["timestamp"] = pd.to_datetime(cpu_df["timestamp"], unit="s")
                monitoring_data["cpu"] = cpu_df

            # Load GPU power metrics
            if experiment.gpu_power_file and experiment.gpu_power_file.exists():
                power_df = pd.read_csv(experiment.gpu_power_file)
                power_df["timestamp"] = pd.to_datetime(power_df["timestamp"], unit="s")
                monitoring_data["gpu_power"] = power_df

            # Load GPU temperature metrics
            if experiment.gpu_temp_file and experiment.gpu_temp_file.exists():
                temp_df = pd.read_csv(experiment.gpu_temp_file)
                temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], unit="s")
                monitoring_data["gpu_temp"] = temp_df

        except Exception as e:
            logger.error(f"Error loading monitoring data for {experiment.result_file.name}: {e}")

        return monitoring_data

    def _calculate_monitoring_summary(
        self, experiment: ExperimentFiles, monitoring_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for monitoring data."""
        summary: Dict[str, Any] = {"experiment": experiment.result_file.name}

        # Calculate experiment duration from monitoring data
        duration = self._estimate_duration_from_monitoring(monitoring_data)
        summary["duration_seconds"] = duration

        # CPU metrics summary
        if "cpu" in monitoring_data:
            cpu_df = monitoring_data["cpu"]
            summary.update(
                {
                    "avg_cpu_usage": (
                        cpu_df["cpu_user_percent"] + cpu_df["cpu_system_percent"]
                    ).mean(),
                    "max_load_avg": cpu_df["load_avg_1min"].max(),
                    "cpu_stability": (
                        cpu_df["cpu_user_percent"] + cpu_df["cpu_system_percent"]
                    ).std(),
                    "avg_cpu_idle": cpu_df["cpu_idle_percent"].mean(),
                }
            )

        # GPU power summary
        if "gpu_power" in monitoring_data:
            power_df = monitoring_data["gpu_power"]
            # Filter out any malformed data
            power_df_clean = power_df[power_df["power_watts"] > 0]
            if not power_df_clean.empty:
                # Calculate total power across all cards per timestamp
                total_power_series = power_df_clean.groupby("timestamp")["power_watts"].sum()
                summary.update(
                    {
                        "avg_total_power": total_power_series.mean(),
                        "max_total_power": total_power_series.max(),
                        "avg_per_gpu_power": power_df_clean.groupby("device")["power_watts"]
                        .mean()
                        .mean(),
                        "power_stability": total_power_series.std(),
                        "num_gpus_monitored": power_df_clean["device"].nunique(),
                    }
                )

        # GPU temperature summary
        if "gpu_temp" in monitoring_data:
            temp_df = monitoring_data["gpu_temp"]
            summary.update(
                {
                    "max_gpu_temp_edge": temp_df["temp_edge_celsius"].max(),
                    "max_gpu_temp_junction": temp_df["temp_junction_celsius"].max(),
                    "avg_gpu_temp_edge": temp_df["temp_edge_celsius"].mean(),
                    "avg_gpu_temp_junction": temp_df["temp_junction_celsius"].mean(),
                    # AMD MI300X thermal throttling typically occurs around 90-95°C junction temp
                    "thermal_throttling_risk": temp_df["temp_junction_celsius"].max() > 90.0,
                }
            )

        return summary

    @staticmethod
    def _estimate_duration_from_monitoring(monitoring_data: Dict[str, pd.DataFrame]) -> float:
        """Estimate experiment duration from monitoring data timestamps."""
        duration = 0.0

        # Try CPU data first (most consistent)
        if "cpu" in monitoring_data:
            cpu_df = monitoring_data["cpu"]
            if len(cpu_df) > 1:
                duration = cpu_df["timestamp"].max() - cpu_df["timestamp"].min()
                return cast(float, duration.total_seconds())

        # Fallback to GPU power data
        if "gpu_power" in monitoring_data:
            power_df = monitoring_data["gpu_power"]
            if len(power_df) > 1:
                duration_series = power_df["timestamp"].max() - power_df["timestamp"].min()
                return cast(float, duration_series.total_seconds())

        # Fallback to GPU temperature data
        if "gpu_temp" in monitoring_data:
            temp_df = monitoring_data["gpu_temp"]
            if len(temp_df) > 1:
                duration_series = temp_df["timestamp"].max() - temp_df["timestamp"].min()
                return cast(float, duration_series.total_seconds())

        return 0.0

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
        """Extract metrics from vLLM latency benchmark results.

        This method processes vLLM latency benchmark data and calculates standardized
        performance metrics. It handles the important distinction between per-request
        completion rate and system-level throughput for batch processing scenarios.

        **Key Metric Definitions:**

        - **Per-Request Completion Rate**: 1 / avg_latency (requests/second/experiment)
          - Measures how frequently individual requests complete
          - Lower values for larger batch sizes due to queueing delays

        - **Batch-Level Throughput**: batch_size / avg_latency (theoretical max)
          - Measures aggregate system processing capacity
          - Not calculated here as batch_size is not available in this context
          - Calculated separately at the experiment level

        **Why Per-Request Rates Appear "Low" for Large Batches:**

        For batch sizes >1, individual requests experience queueing delays:
        - Batch size 1: Request processes immediately → high completion rate
        - Batch size 32: Each request waits for 31 others → low completion rate
        - This reflects real-world user experience in interactive systems

        Args:
            data (Dict[str, Any]): Raw vLLM benchmark results containing:
                - avg_latency: Mean response time across all requests (seconds)
                - latencies: List of individual request latencies (optional)
                - percentiles: Dictionary of latency percentiles (optional)

        Returns:
            BenchmarkMetrics: Standardized metrics object with:
                - avg_latency: Average request latency (seconds)
                - throughput: Per-request completion rate (requests/second)
                - percentile metrics: P50, P90, P95, P99 latencies
                - statistical measures: Standard deviation, iteration count

        Raises:
            KeyError: If required 'avg_latency' field is missing from data
            ValueError: If avg_latency is zero or negative

        Example:
            ```
            # Sample vLLM latency data
            vllm_data = {
                "avg_latency": 2.5,  # seconds per request
                "latencies": [2.3, 2.5, 2.7, 2.4, 2.6],
                "percentiles": {"50": 2.5, "90": 2.7, "95": 2.8, "99": 2.9}
            }

            metrics = BenchmarkAnalyzer._extract_vllm_latency_metrics(vllm_data)

            # Results interpretation:
            print(f"Request completion rate: {metrics.throughput:.3f} req/s")
            # Output: "Request completion rate: 0.400 req/s"

            # For batch size 8, system throughput would be:
            # system_throughput = 8 * metrics.throughput = 3.200 req/s
            # (calculated at experiment level with batch size context)
            ```

        Note:
            This method focuses on per-request performance metrics consistent with
            latency benchmarking standards. System-level throughput calculations
            requiring batch size context are handled in the experiment analysis layer.
        """

        # Validate required fields
        avg_latency = data.get("avg_latency")
        if avg_latency is None:
            raise KeyError("Required field 'avg_latency' missing from benchmark data")
        if avg_latency <= 0:
            raise ValueError(f"Invalid avg_latency value: {avg_latency}. Must be positive.")

        # Extract optional data with robust handling
        latencies = data.get("latencies", [])
        percentiles = data.get("percentiles", {})

        # Calculate statistical measures
        latency_std = 0.0
        total_iterations = len(latencies)

        if len(latencies) > 1:
            import statistics

            try:
                latency_std = statistics.stdev(latencies)
            except statistics.StatisticsError:
                logger.warning("Could not calculate latency standard deviation")
                latency_std = 0.0

        # Extract percentiles with flexible key formats
        # vLLM may use string keys ("50") or integer keys (50)
        def get_percentile(p: int) -> float:
            """Extract percentile with flexible key handling."""
            return float(percentiles.get(str(p), percentiles.get(p, 0.0)))

        # Calculate request completion rate (requests per second for single request processing)
        # This represents the inverse of latency: how frequently one request completes
        throughput = 1.0 / avg_latency

        return BenchmarkMetrics(
            # Core latency metrics
            avg_latency=avg_latency,
            latency_std=latency_std,
            # Percentile latencies (in seconds)
            p50_latency=get_percentile(50),
            p90_latency=get_percentile(90),
            p95_latency=get_percentile(95),
            p99_latency=get_percentile(99),
            # Per-request completion rate (requests/second per experiment)
            # Note: This is NOT system-level throughput for batch processing
            throughput=throughput,
            # Token-level metrics (not available in latency-only benchmarks)
            tokens_per_second=0.0,
            # Experimental metadata
            total_iterations=total_iterations if total_iterations > 0 else 1,
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


class BatchEfficiencyAnalyzer:
    """Analyze batch efficiency across multiple batch size configurations."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.by_batch_size = self._group_by_batch_size()

    def _group_by_batch_size(self) -> Dict[int, List[BenchmarkResult]]:
        """Group results by batch size for comparison."""
        from collections import defaultdict

        groups: Dict[int, List[BenchmarkResult]] = defaultdict(list)
        for result in self.results:
            groups[result.config.batch_size].append(result)
        return dict(groups)

    def calculate_scaling_efficiency(self, baseline_batch_size: int = 1) -> Dict[int, float]:
        """
        Calculate how efficiently each batch size scales compared to baseline.

        Returns efficiency ratios where:
        - 1.0 = same efficiency as baseline
        - >1.0 = better than baseline
        - <1.0 = worse than baseline
        """
        if baseline_batch_size not in self.by_batch_size:
            raise ValueError(f"No data for baseline batch size {baseline_batch_size}")

        baseline_results = self.by_batch_size[baseline_batch_size]
        baseline_throughput = sum(r.metrics.throughput for r in baseline_results) / len(
            baseline_results
        )

        efficiencies = {}
        for batch_size, results in self.by_batch_size.items():
            avg_system_throughput = sum(r.system_throughput for r in results) / len(results)
            theoretical_throughput = batch_size * baseline_throughput
            efficiencies[batch_size] = avg_system_throughput / theoretical_throughput

        return efficiencies

    def get_scaling_grades(self, baseline_batch_size: int = 1) -> Dict[int, str]:
        """Generates a human-readable performance grade for each batch size."""

        efficiency_ratios = self.calculate_scaling_efficiency(baseline_batch_size)
        grades = {}

        for batch_size, ratio in efficiency_ratios.items():
            if ratio >= 1.1:
                grades[batch_size] = "A+ (Excellent)"
            elif ratio >= 1.0:
                grades[batch_size] = "A (Very Good)"
            elif ratio >= 0.9:
                grades[batch_size] = "B (Good)"
            elif ratio >= 0.8:
                grades[batch_size] = "C (Fair)"
            elif ratio >= 0.7:
                grades[batch_size] = "D (Poor)"
            else:
                grades[batch_size] = "F (Very Poor)"

        return grades
