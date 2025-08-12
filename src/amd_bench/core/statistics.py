"""Statistical analysis for benchmark results."""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..schemas.benchmark import BenchmarkResult
from ..utils.logging import get_logger
from ..utils.paths import ensure_directory

logger = get_logger(__name__)


class StatisticalAnalyzer:
    """Generates statistical summaries of benchmark results."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.df = self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        records = []
        for result in self.results:
            records.append(
                {
                    "file": result.file,
                    "experiment_id": result.experiment_id,
                    "model_short_name": result.model_short_name,
                    "model": result.config.model,
                    "benchmark_type": result.config.benchmark_type,
                    "batch_size": result.config.batch_size,
                    "input_length": result.config.input_length,
                    "output_length": result.config.output_length,
                    "dtype": result.config.dtype,
                    "memory_util": result.config.memory_util,
                    "timestamp": result.config.timestamp,
                    "avg_latency": result.metrics.avg_latency,
                    "latency_std": result.metrics.latency_std,
                    "p50_latency": result.metrics.p50_latency,
                    "p90_latency": result.metrics.p90_latency,
                    "p95_latency": result.metrics.p95_latency,
                    "p99_latency": result.metrics.p99_latency,
                    "throughput": result.metrics.throughput,
                    "tokens_per_second": result.metrics.tokens_per_second,
                    "total_iterations": result.metrics.total_iterations,
                    "efficiency_score": result.efficiency_score,
                }
            )

        return pd.DataFrame(records)

    def generate_model_summary(self) -> pd.DataFrame:
        """Generate summary statistics grouped by model."""
        if self.df.empty:
            return pd.DataFrame()

        summary = (
            self.df.groupby(["model", "benchmark_type"])
            .agg(
                {
                    "avg_latency": ["mean", "std", "min", "max"],
                    "p90_latency": ["mean", "std"],
                    "p99_latency": ["mean", "std"],
                    "throughput": ["mean", "std", "min", "max"],
                    "batch_size": "count",
                }
            )
            .round(4)
        )

        summary.columns = ["_".join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        summary.rename(columns={"batch_size_count": "num_experiments"}, inplace=True)

        return summary

    def generate_batch_size_summary(self) -> pd.DataFrame:
        """Generate summary statistics grouped by batch size."""
        if self.df.empty:
            return pd.DataFrame()

        summary = (
            self.df.groupby("batch_size")
            .agg(
                {
                    "avg_latency": ["mean", "std"],
                    "throughput": ["mean", "std"],
                    "efficiency_score": ["mean", "std"],
                    "model": "count",
                }
            )
            .round(4)
        )

        summary.columns = ["_".join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        summary.rename(columns={"model_count": "num_experiments"}, inplace=True)

        return summary

    def generate_memory_util_summary(self) -> pd.DataFrame:
        """Generate summary statistics grouped by memory utilization."""
        if self.df.empty:
            return pd.DataFrame()

        summary = (
            self.df.groupby("memory_util")
            .agg({"avg_latency": ["mean", "std", "count"], "throughput": ["mean", "std"]})
            .round(4)
        )

        summary.columns = ["_".join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        summary.rename(columns={"avg_latency_count": "num_experiments"}, inplace=True)

        return summary

    def export_summaries(self, output_dir: Path) -> Dict[str, Path]:
        """Export all summary tables to CSV files."""
        tables_dir = ensure_directory(output_dir / "tables")

        saved_files = {}

        # Model summary
        model_summary = self.generate_model_summary()
        if not model_summary.empty:
            model_file = tables_dir / "model_performance_summary.csv"
            model_summary.to_csv(model_file, index=False)
            saved_files["model_summary"] = model_file

        # Batch size summary
        batch_summary = self.generate_batch_size_summary()
        if not batch_summary.empty:
            batch_file = tables_dir / "batch_size_analysis.csv"
            batch_summary.to_csv(batch_file, index=False)
            saved_files["batch_summary"] = batch_file

        # Memory utilization summary
        memory_summary = self.generate_memory_util_summary()
        if not memory_summary.empty:
            memory_file = tables_dir / "memory_utilization_analysis.csv"
            memory_summary.to_csv(memory_file, index=False)
            saved_files["memory_summary"] = memory_file

        # Raw data export
        if not self.df.empty:
            raw_file = tables_dir / "raw_results.csv"
            self.df.to_csv(raw_file, index=False)
            saved_files["raw_data"] = raw_file

        logger.info(f"Exported {len(saved_files)} summary tables to {tables_dir}")
        return saved_files
