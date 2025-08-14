"""Report generation for benchmark analysis results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import pandas as pd

from amd_bench.schemas.benchmark import AnalysisConfig, BenchmarkResult
from amd_bench.utils.logging import get_logger
from amd_bench.utils.paths import ensure_directory

logger = get_logger(__name__)


class ReportGenerator:
    """Generates analysis reports in multiple formats.

    This class creates comprehensive reports from benchmark analysis results, supporting
    both human-readable markdown reports and machine-readable JSON summaries. It handles
    statistical formatting, and monitoring data integration.

    The ReportGenerator produces:
    - **Markdown Reports**: Detailed analysis with tables, insights, and recommendations
    - **JSON Summaries**: Structured data for programmatic consumption
    - **Executive Summaries**: High-level performance insights
    - **Configuration Analysis**: Optimal parameter recommendations

    Attributes:
        config (AnalysisConfig): Configuration settings for analysis scope and format.
        results (List[BenchmarkResult]): Processed benchmark results data.
        stats_analyzer (StatisticalAnalyzer): Statistical analysis engine for summaries.

    Example:
        ```
        # Basic usage
        config = AnalysisConfig(input_dir=Path("data"), output_dir=Path("output"))
        results = analyzer.load_results()
        stats = StatisticalAnalyzer(results)

        generator = ReportGenerator(config, results, stats)
        reports = generator.create_reports(Path("reports"))

        print(f"Generated reports: {list(reports.keys())}")
        # Output: ['markdown', 'json']

        # Access generated files
        markdown_report = reports['markdown']
        json_summary = reports['json']
        ```

    Note:
        Report generation scales with dataset size. Large datasets with monitoring
        data may require additional processing time for comprehensive analysis.
        All reports include metadata about generation time and data sources.
    """

    def __init__(
        self,
        config: AnalysisConfig,
        results: List[BenchmarkResult],
        stats_analyzer: Optional[object] = None,
    ):
        """Initialize report generator."""
        self.config = config
        self.results = results
        self.stats_analyzer = stats_analyzer

    def create_reports(self, output_dir: Path) -> Dict[str, Path]:
        """Create all report formats."""
        output_dir = ensure_directory(output_dir)

        generated_reports = {}

        # Markdown report
        markdown_path = output_dir / "benchmark_analysis_report.md"
        self._create_markdown_report(markdown_path)
        generated_reports["markdown"] = markdown_path

        # JSON summary
        json_path = output_dir / "analysis_summary.json"
        self._create_json_summary(json_path)
        generated_reports["json"] = json_path

        logger.info(f"Generated {len(generated_reports)} reports in {output_dir}")
        return generated_reports

    def _create_markdown_report(self, path: Path) -> None:
        """Create comprehensive markdown analysis report."""
        with open(path, "w", encoding="utf-8") as f:
            # Header
            f.write("# Benchmark Analysis Report\n")
            f.write(f"**Generated**: {self._get_current_timestamp()}\n")
            f.write(f"**Total Results**: {len(self.results)}\n")
            f.write(f"**Input Directory**: {self.config.input_dir}\n\n")

            if not self.results:
                f.write("No benchmark results available for analysis.\n")
                return

            # Executive Summary
            f.write("## Executive Summary\n\n")
            self._write_executive_summary(f)

            # Model Performance
            f.write("\n## Model Performance Overview\n\n")
            self._write_model_performance_section(f)

            # Configuration Analysis
            f.write("\n## Configuration Analysis\n\n")
            self._write_configuration_analysis(f)

            # Key Findings
            f.write("\n## Key Findings\n\n")
            self._write_key_findings(f)

            # System Monitoring Analysis
            if self.config.include_monitoring_data:
                monitoring_data = self._load_monitoring_data()
                if monitoring_data:
                    f.write("## System Monitoring Analysis\n\n")
                    self._write_monitoring_analysis_section(
                        f,
                        monitoring_data["monitoring_summaries"],
                        monitoring_data["thermal_analysis"],
                        monitoring_data["power_analysis"],
                    )

        logger.info(f"Markdown report created: {path}")

    def _load_monitoring_data(self) -> Dict[str, pd.DataFrame]:
        """Load monitoring data from existing CSV files."""
        monitoring_data = {}
        tables_dir = self.config.output_dir / "tables"

        try:
            # Load monitoring summaries
            monitoring_file = tables_dir / "monitoring_summary.csv"
            if monitoring_file.exists():
                monitoring_data["monitoring_summaries"] = pd.read_csv(monitoring_file)

            # Load thermal analysis
            thermal_file = tables_dir / "thermal_analysis.csv"
            if thermal_file.exists():
                monitoring_data["thermal_analysis"] = pd.read_csv(thermal_file)

            # Load power analysis
            power_file = tables_dir / "power_analysis.csv"
            if power_file.exists():
                monitoring_data["power_analysis"] = pd.read_csv(power_file)

            logger.info(f"Loaded {len(monitoring_data)} monitoring datasets")

        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")

        return monitoring_data

    def _write_executive_summary(self, file: TextIO) -> None:
        """Write executive summary section."""
        models = {r.model_short_name for r in self.results}
        avg_latency = sum(r.metrics.avg_latency for r in self.results) / len(self.results)
        avg_throughput = sum(r.metrics.throughput for r in self.results) / len(self.results)

        file.write(
            f"This analysis covers **{len(models)} models** across **{len(self.results)} experiments**.\n\n"
        )
        file.write(f"- **Average Latency**: {avg_latency:.4f} seconds\n")
        file.write(f"- **Average Throughput**: {avg_throughput:.2f} requests/second\n")
        file.write(f"- **Models Tested**: {', '.join(sorted(models))}\n")

    def _write_model_performance_section(self, file: TextIO) -> None:
        """Write model performance section to markdown report."""
        if not self.results:
            file.write("No performance data available.\n\n")
            return

        # Calculate performance statistics by model
        models: Dict[str, Any] = {}
        for result in self.results:
            model_name = result.model_short_name
            if model_name not in models:
                models[model_name] = {"latencies": [], "throughputs": [], "efficiency_scores": []}

            models[model_name]["latencies"].append(result.metrics.avg_latency)
            models[model_name]["throughputs"].append(result.metrics.throughput)
            models[model_name]["efficiency_scores"].append(result.efficiency_score)

        file.write("### Performance Summary by Model\n\n")
        file.write("| Model | Avg Latency (s) | Avg Throughput (req/s) | Efficiency Score |\n")
        file.write("|-------|-----------------|------------------------|------------------|\n")

        for model_name, stats in models.items():
            avg_latency = sum(stats["latencies"]) / len(stats["latencies"])
            avg_throughput = sum(stats["throughputs"]) / len(stats["throughputs"])
            avg_efficiency = sum(stats["efficiency_scores"]) / len(stats["efficiency_scores"])

            file.write(
                f"| {model_name} | {avg_latency:.4f} | {avg_throughput:.2f} | {avg_efficiency:.2f} |\n"
            )

        file.write("\n")

    def _write_configuration_analysis(self, file: TextIO) -> None:
        """Write configuration analysis section."""
        if not self.results:
            file.write("No configuration data available.\n\n")
            return

        # Analyze configuration patterns
        batch_sizes = {r.config.batch_size for r in self.results}
        memory_utils = {r.config.memory_util for r in self.results}
        dtypes = {r.config.dtype for r in self.results}

        file.write("### Configuration Parameters Tested\n\n")
        file.write(f"- **Batch Sizes**: {sorted(batch_sizes)}\n")
        file.write(f"- **Memory Utilizations**: {sorted(memory_utils)}\n")
        file.write(f"- **Data Types**: {sorted(dtypes)}\n")
        file.write(f"- **Total Configurations**: {len(self.results)}\n\n")

        # Find optimal configurations
        if self.results:
            best_latency = min(self.results, key=lambda r: r.metrics.avg_latency)
            best_throughput = max(self.results, key=lambda r: r.metrics.throughput)

            file.write("### Optimal Configurations\n")
            file.write(f"- **Best Latency**: {best_latency.model_short_name} ")
            file.write(
                f"(bs={best_latency.config.batch_size}, mem={best_latency.config.memory_util}) "
            )
            file.write(f"- {best_latency.metrics.avg_latency:.4f}s\n")

            file.write(f"- **Best Throughput**: {best_throughput.model_short_name} ")
            file.write(
                f"(bs={best_throughput.config.batch_size}, mem={best_throughput.config.memory_util}) "
            )
            file.write(f"- {best_throughput.metrics.throughput:.2f} req/s\n")

    def _write_key_findings(self, file: TextIO) -> None:
        """Write key findings section."""
        if not self.results:
            file.write("No data available for analysis.\n\n")
            return

        # Performance insights
        models = {r.model_short_name for r in self.results}
        avg_latency = sum(r.metrics.avg_latency for r in self.results) / len(self.results)
        avg_throughput = sum(r.metrics.throughput for r in self.results) / len(self.results)

        file.write(f"1. **Models Evaluated**: {len(models)} different models tested\n")
        file.write(
            f"2. **Average Performance**: {avg_latency:.4f}s latency, {avg_throughput:.2f} req/s throughput\n"
        )

        # Batch size impact analysis
        batch_impact: Dict[int, List[float]] = {}
        for result in self.results:
            bs = result.config.batch_size
            if bs not in batch_impact:
                batch_impact[bs] = []
            batch_impact[bs].append(result.metrics.avg_latency)

        if len(batch_impact) > 1:
            best_bs = min(
                batch_impact.keys(), key=lambda x: sum(batch_impact[x]) / len(batch_impact[x])
            )
            file.write(f"3. **Optimal Batch Size**: {best_bs} showed best average performance\n")

        # Memory utilization insights
        mem_utils = [r.config.memory_util for r in self.results]
        if mem_utils:
            avg_mem_util = sum(mem_utils) / len(mem_utils)
            file.write(f"4. **Memory Utilization**: Average {avg_mem_util:.2f} across all tests\n")

        file.write("\n")

    @staticmethod
    def _write_monitoring_analysis_section(
        file: TextIO,
        monitoring_summaries: pd.DataFrame,
        thermal_analysis: pd.DataFrame,
        power_analysis: pd.DataFrame,
    ) -> None:
        """Write comprehensive monitoring analysis section."""

        if monitoring_summaries.empty and thermal_analysis.empty and power_analysis.empty:
            return

        # Power efficiency insights
        if not power_analysis.empty:
            avg_power = power_analysis["avg_total_power"].mean()
            max_power = power_analysis["max_total_power"].max()
            avg_efficiency = power_analysis["power_efficiency"].mean()
            power_stability = power_analysis["power_stability"].std()

            file.write("### Power Consumption Analysis\n\n")
            file.write(
                f"- **Average Total Power Consumption**: {avg_power:.1f}W across 8x MI300X GPUs\n"
            )
            file.write(f"  - **Peak Power Draw**: {max_power:.1f}W\n")
            file.write(f"- **Per-GPU Efficiency (avg)**: {avg_efficiency:.1f}W\n")
            file.write(f"- **Power Stability**: {power_stability:.2f}W variation\n\n")

        # Thermal analysis
        if not thermal_analysis.empty:
            max_edge_temp = thermal_analysis["max_edge_temp"].max()
            avg_edge_temp = thermal_analysis["avg_edge_temp"].mean()
            max_junction_temp = thermal_analysis["max_junction_temp"].max()
            thermal_risk_count = thermal_analysis["thermal_throttling_risk"].sum()

            file.write("### Thermal Performance\n\n")

            edge_temperature_note = (
                "(excellent cooling ✅)" if max_edge_temp < max_junction_temp else ""
            )
            file.write(f"- **Peak Edge Temperature**: {max_edge_temp}°C {edge_temperature_note}\n")

            temperature_headroom = 90 - max_junction_temp
            junction_temperature_note = (
                f"{temperature_headroom}°C below throttling threshold ✅"
                if temperature_headroom > 0
                else "⚠️ throttling threshold exceeded. Please investigate ⚠️"
            )
            file.write(
                f"- **Peak Junction Temperature**: {max_junction_temp}°C ({junction_temperature_note})\n"
            )

            file.write(f"- **Average Operating Temperature**: {avg_edge_temp:.1f}°C (edge)\n")
            thermal_note = (
                "**No thermal throttling events across all tests** ✅\n\n"
                if thermal_risk_count == 0
                else f"**Thermal Throttling Events**: {thermal_risk_count} out of {len(thermal_analysis)} experiments ⚠️\n\n"
            )
            file.write(f"- {thermal_note}\n")

            std_temp_stability = thermal_analysis["temp_stability"].std()
            file.write(f"- **Temperature stability**: {std_temp_stability:.1f}°C\n")

            file.write(
                """> **Note**: Edge vs. junction temperatures are different sensors. Junction temperature is
            typically 5-15°C higher than edge temperature and is the critical metric for throttling decisions.\n"""
            )

        # System stability from monitoring summaries
        avg_cpu_usage = monitoring_summaries["avg_cpu_usage"].mean()
        avg_duration = monitoring_summaries["duration_seconds"].mean()
        cpu_stability = monitoring_summaries["cpu_stability"].std()

        file.write("### System Stability\n\n")
        if avg_cpu_usage < 15:
            avg_cpu_usage_note = "confirms GPU-bound workloads ✅"
        elif avg_cpu_usage < 50:
            avg_cpu_usage_note = ">15% CPU usage during pure inference workloads indicates inefficient GPU utilization ⚠️"
        else:
            avg_cpu_usage_note = (
                ">50% CPU usage with low GPU utilization is a clear indication of CPU bottleneck ⚠️"
            )
        file.write(f"- **CPU Utilization (avg)**: {avg_cpu_usage:.2f}% ({avg_cpu_usage_note})\n")
        file.write(f"- **Experiment Duration (avg)**: {avg_duration/60:.1f} minutes\n")
        file.write(f"- **CPU Load Stability**: {cpu_stability:.2f}% variation\n")

    def _create_json_summary(self, path: Path) -> None:
        """Create JSON summary of analysis results."""
        if not self.results:
            logger.warning("No results available for JSON summary.")
            return

        models = list({r.model_short_name for r in self.results})

        summary = {
            "metadata": {
                "generated_at": self._get_current_timestamp(),
                "total_results": len(self.results),
                "input_directory": str(self.config.input_dir),
                "output_directory": str(self.config.output_dir),
            },
            "models": models,
            "performance_metrics": {
                "avg_latency": {
                    "mean": sum(r.metrics.avg_latency for r in self.results) / len(self.results),
                    "min": min(r.metrics.avg_latency for r in self.results),
                    "max": max(r.metrics.avg_latency for r in self.results),
                },
                "throughput": {
                    "mean": sum(r.metrics.throughput for r in self.results) / len(self.results),
                    "min": min(r.metrics.throughput for r in self.results),
                    "max": max(r.metrics.throughput for r in self.results),
                },
            },
            "configuration_summary": {
                "batch_sizes": sorted({r.config.batch_size for r in self.results}),
                "memory_utilizations": sorted({r.config.memory_util for r in self.results}),
                "data_types": sorted({r.config.dtype for r in self.results}),
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"JSON summary created: {path}")

    @staticmethod
    def _get_current_timestamp() -> str:
        """Get current timestamp for reports."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
