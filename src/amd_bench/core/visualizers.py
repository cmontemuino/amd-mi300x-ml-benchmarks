"""Visualization generation for benchmark analysis."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..schemas.benchmark import AnalysisConfig, BenchmarkResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkVisualizer:
    """Generate publication-quality visualizations for AMD MI300X benchmark analysis."""

    def __init__(self, results: List[BenchmarkResult], config: AnalysisConfig):
        """Initialize visualizer with benchmark results and configuration."""
        self.results = results
        self.config = config

        # Convert results to DataFrame for easier plotting
        self.df = self._create_dataframe()

        # Set up plotting style for publication quality
        self._setup_plotting_style()

        logger.info(f"Visualizer initialized with {len(self.results)} benchmark results")

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert benchmark results to pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()

        records = []
        for result in self.results:
            record = {
                # Metadata
                "file": result.file,
                "experiment_id": result.experiment_id,
                "model_short_name": result.model_short_name,
                # Configuration
                "model": result.config.model,
                "benchmark_type": result.config.benchmark_type,
                "batch_size": result.config.batch_size,
                "input_length": result.config.input_length,
                "output_length": result.config.output_length,
                "dtype": result.config.dtype,
                "memory_util": result.config.memory_util,
                "timestamp": result.config.timestamp,
                # Metrics
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
            records.append(record)

        return pd.DataFrame(records)

    def _setup_plotting_style(self) -> None:
        """Configure matplotlib and seaborn for publication-quality plots."""
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # Configure matplotlib for high-quality output
        plt.rcParams.update(
            {
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.format": "png",
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "figure.titlesize": 18,
                "legend.fontsize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
            }
        )

    def create_all_plots(self, output_dir: Path) -> Dict[str, Path]:
        """Generate all visualization plots and return file paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.df.empty:
            logger.warning("No data available for visualization")
            return {}

        plot_files = {}

        try:
            # Core performance plots
            plot_files.update(self._create_latency_analysis(output_dir))
            plot_files.update(self._create_throughput_analysis(output_dir))
            plot_files.update(self._create_batch_size_scaling(output_dir))
            plot_files.update(self._create_memory_efficiency(output_dir))
            plot_files.update(self._create_batch_memory_interaction(output_dir))

            logger.info(f"Generated {len(plot_files)} visualization plots")

        except Exception as e:
            logger.error(f"Error generating plots: {e}")

        return plot_files

    def _create_latency_analysis(self, output_dir: Path) -> Dict[str, Path]:
        """Create comprehensive latency analysis plots."""
        if self.df.empty:
            return {}

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("AMD MI300X Latency Performance Analysis", fontsize=16, fontweight="bold")

        # Average latency by model
        sns.boxplot(data=self.df, x="model", y="avg_latency", ax=axes[0, 0])
        axes[0, 0].set_title("Average Latency by Model")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # P90 vs P99 latency
        sns.scatterplot(
            data=self.df,
            x="p90_latency",
            y="p99_latency",
            hue="model",
            size="batch_size",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("P90 vs P99 Latency")

        # Batch size impact
        sns.lineplot(
            data=self.df, x="batch_size", y="avg_latency", hue="model", marker="o", ax=axes[1, 0]
        )
        axes[1, 0].set_title("Batch Size Impact on Latency")
        try:
            # Latency distribution
            for model in self.df["model"].unique():
                model_data = self.df[self.df["model"] == model]
                axes[1, 1].hist(model_data["avg_latency"], alpha=0.7, label=model, bins=20)
            axes[1, 1].set_title("Latency Distribution by Model")
            axes[1, 1].set_xlabel("Average Latency (s)")
            axes[1, 1].legend()

            plt.tight_layout()
            plot_file = output_dir / "latency_analysis.png"
            plt.savefig(plot_file)
            plt.close()

            logger.info(f"Created latency analysis plot: {plot_file}")
            return {"latency_analysis": plot_file}

        except Exception as e:
            logger.error(f"Error creating latency analysis: {e}")
            plt.close()
            return {}

    def _create_throughput_analysis(self, output_dir: Path) -> Dict[str, Path]:
        """Create throughput comparison visualizations."""
        if self.df.empty:
            return {}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("AMD MI300X Throughput Performance Analysis", fontsize=16, fontweight="bold")

        try:
            # 1. Throughput by model and batch size
            pivot_data = self.df.pivot_table(
                values="throughput", index="model_short_name", columns="batch_size", aggfunc="mean"
            )

            if not pivot_data.empty:
                pivot_data.plot(kind="bar", ax=ax1, rot=45)
                ax1.set_title("Throughput by Model and Batch Size")
                ax1.set_xlabel("Model")
                ax1.set_ylabel("Throughput (requests/second)")
                ax1.legend(title="Batch Size")

            # 2. Memory utilization vs throughput
            memory_grouped = (
                self.df.groupby(["memory_util", "model_short_name"]).throughput.mean().reset_index()
            )

            for model in memory_grouped["model_short_name"].unique():
                model_data = memory_grouped[memory_grouped["model_short_name"] == model]
                ax2.plot(
                    model_data["memory_util"],
                    model_data["throughput"],
                    marker="o",
                    label=model,
                    linewidth=2,
                    markersize=8,
                )

            ax2.set_xlabel("Memory Utilization")
            ax2.set_ylabel("Throughput (requests/second)")
            ax2.set_title("Memory Utilization vs Throughput")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_file = output_dir / "throughput_comparison.png"
            plt.savefig(plot_file)
            plt.close()

            logger.info(f"Created throughput analysis plot: {plot_file}")
            return {"throughput_comparison": plot_file}

        except Exception as e:
            logger.error(f"Error creating throughput analysis: {e}")
            plt.close()
            return {}

    def _create_batch_size_scaling(self, output_dir: Path) -> Dict[str, Path]:
        """Create batch size scaling analysis plots."""
        if self.df.empty:
            return {}

        # Aggregate data by model and batch size
        df_agg = (
            self.df.groupby(["model_short_name", "batch_size"])
            .agg({"avg_latency": "mean", "throughput": "mean"})
            .reset_index()
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("AMD MI300X Batch Size Scaling Analysis", fontsize=16, fontweight="bold")

        try:
            # 1. Latency scaling
            for model in df_agg["model_short_name"].unique():
                model_data = df_agg[df_agg["model_short_name"] == model].sort_values("batch_size")
                ax1.plot(
                    model_data["batch_size"],
                    model_data["avg_latency"],
                    marker="o",
                    label=model,
                    linewidth=2,
                    markersize=8,
                )

            ax1.set_xlabel("Batch Size")
            ax1.set_ylabel("Average Latency (s)")
            ax1.set_title("Latency vs Batch Size (averaged across memory utilizations)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Throughput scaling
            for model in df_agg["model_short_name"].unique():
                model_data = df_agg[df_agg["model_short_name"] == model].sort_values("batch_size")
                ax2.plot(
                    model_data["batch_size"],
                    model_data["throughput"],
                    marker="s",
                    label=model,
                    linewidth=2,
                    markersize=8,
                )

            ax2.set_xlabel("Batch Size")
            ax2.set_ylabel("Throughput (req/s)")
            ax2.set_title("Throughput vs Batch Size (averaged across memory utilizations")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_file = output_dir / "batch_size_scaling.png"
            plt.savefig(plot_file)
            plt.close()

            logger.info(f"Created batch size scaling plot: {plot_file}")
            return {"batch_size_scaling": plot_file}

        except Exception as e:
            logger.error(f"Error creating batch size scaling plot: {e}")
            plt.close()
            return {}

    def _create_memory_efficiency(self, output_dir: Path) -> Dict[str, Path]:
        """Create memory efficiency analysis plots."""
        if self.df.empty:
            return {}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("AMD MI300X Memory Efficiency Analysis", fontsize=18, fontweight="bold")

        try:
            # 1. Memory utilization vs efficiency scatter plot
            scatter = ax1.scatter(
                self.df["memory_util"],
                self.df["efficiency_score"],
                c=self.df["avg_latency"],
                s=self.df["batch_size"] * 20,
                alpha=0.6,
                cmap="viridis_r",  # Reverse colormap so lower latency is brighter
            )
            ax1.set_xlabel("Memory Utilization")
            ax1.set_ylabel("Efficiency Score (throughput/latency)")
            ax1.set_title("Memory Utilization vs Efficiency")
            ax1.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label("Average Latency (seconds)")

            # 2. Efficiency trends by memory utilization
            df_memory_agg = (
                self.df.groupby(["model_short_name", "memory_util"])
                .agg({"efficiency_score": "mean"})
                .reset_index()
            )

            for model in df_memory_agg["model_short_name"].unique():
                model_data = df_memory_agg[df_memory_agg["model_short_name"] == model].sort_values(
                    "memory_util"
                )
                ax2.plot(
                    model_data["memory_util"],
                    model_data["efficiency_score"],
                    marker="o",
                    label=model,
                    linewidth=2,
                    markersize=8,
                )

            ax2.set_xlabel("Memory Utilization")
            ax2.set_ylabel("Efficiency Score")
            ax2.set_title("Efficiency Score by Model (averaged across batch sizes)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_file = output_dir / "memory_efficiency.png"
            plt.savefig(plot_file)
            plt.close()

            logger.info(f"Created memory efficiency plot: {plot_file}")
            return {"memory_efficiency": plot_file}

        except Exception as e:
            logger.error(f"Error creating memory efficiency plot: {e}")
            plt.close()
            return {}

    def _create_batch_memory_interaction(self, output_dir: Path) -> Dict[str, Path]:
        """Create plots showing batch size and memory utilization interaction."""
        if self.df.empty:
            return {}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Batch Size Scaling by Memory Utilization", fontsize=16, fontweight="bold")

        try:
            # Plot separate lines for each memory utilization level
            for model in self.df["model_short_name"].unique():
                model_data = self.df[self.df["model_short_name"] == model]
                for mem_util in sorted(model_data["memory_util"].unique()):
                    mem_data = model_data[model_data["memory_util"] == mem_util].sort_values(
                        "batch_size"
                    )

                    if not mem_data.empty:
                        # Latency plot
                        ax1.plot(
                            mem_data["batch_size"],
                            mem_data["avg_latency"],
                            marker="o",
                            label=f"{model} (mem={mem_util})",
                            linewidth=2,
                        )

                        # Throughput plot
                        ax2.plot(
                            mem_data["batch_size"],
                            mem_data["throughput"],
                            marker="s",
                            label=f"{model} (mem={mem_util})",
                            linewidth=2,
                        )

            ax1.set_xlabel("Batch Size")
            ax1.set_ylabel("Average Latency (s)")
            ax1.set_title("Latency vs Batch Size by Memory Utilization")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.set_xlabel("Batch Size")
            ax2.set_ylabel("Throughput (req/s)")
            ax2.set_title("Throughput vs Batch Size by Memory Utilization")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_file = output_dir / "batch_size_scaling_by_memory.png"
            plt.savefig(plot_file)
            plt.close()

            logger.info(f"Created batch-memory interaction plot: {plot_file}")
            return {"batch_size_scaling_by_memory": plot_file}

        except Exception as e:
            logger.error(f"Error creating batch-memory interaction plot: {e}")
            plt.close()
            return {}


class PlotGenerator(BenchmarkVisualizer):
    """Legacy alias for BenchmarkVisualizer to maintain compatibility."""

    def __init__(self, results: List[BenchmarkResult]):
        """Initialize with results only for backward compatibility."""
        # Create a minimal config for compatibility
        from pathlib import Path

        from ..schemas.benchmark import AnalysisConfig

        config = AnalysisConfig(input_dir=Path("."), output_dir=Path("plots"))
        super().__init__(results, config)
