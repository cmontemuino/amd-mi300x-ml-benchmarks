"""Report generation for benchmark analysis results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from ..schemas.benchmark import AnalysisConfig, BenchmarkResult
from ..utils.logging import get_logger
from ..utils.paths import ensure_directory

logger = get_logger(__name__)


class ReportGenerator:
    """Generates analysis reports in multiple formats."""

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
            f.write("# Benchmark Analysis Report\n\n")
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

        logger.info(f"Markdown report created: {path}")

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

            file.write("### Optimal Configurations\n\n")
            file.write(f"**Best Latency**: {best_latency.model_short_name} ")
            file.write(
                f"(bs={best_latency.config.batch_size}, mem={best_latency.config.memory_util}) "
            )
            file.write(f"- {best_latency.metrics.avg_latency:.4f}s\n\n")

            file.write(f"**Best Throughput**: {best_throughput.model_short_name} ")
            file.write(
                f"(bs={best_throughput.config.batch_size}, mem={best_throughput.config.memory_util}) "
            )
            file.write(f"- {best_throughput.metrics.throughput:.2f} req/s\n\n")

    def _write_key_findings(self, file: TextIO) -> None:
        """Write key findings section."""
        if not self.results:
            file.write("No data available for analysis.\n\n")
            return

        file.write("### Key Findings\n\n")

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
