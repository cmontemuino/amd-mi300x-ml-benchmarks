"""Command-line interface for benchmark analysis"""

from pathlib import Path
from typing import Optional

import typer

from amd_bench.core.analysis import BenchmarkAnalyzer
from amd_bench.schemas.benchmark import AnalysisConfig
from amd_bench.utils.paths import load_analysis_config_from_yaml

app = typer.Typer(help="Analysis tools for AMD MI300X benchmarks")


@app.command("run")
def run(
    input_dir: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Input directory containing results",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Output directory for analysis results",
    ),
    logs_dir: Optional[str] = typer.Option(
        "logs",
        help="Sub directory under input-dir for containing experiment logs",
    ),
    monitoring_dir: Optional[str] = typer.Option(
        "monitoring",
        help="Sub directory under input-dir for monitoring metrics",
    ),
    config_file: Optional[str] = typer.Option(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="YAML configuration file path",
    ),
) -> None:
    """Run benchmark analysis with optional YAML configuration"""

    if config_file:
        # Load from YAML configuration
        config = load_analysis_config_from_yaml(config_file)
        typer.echo(f"Loaded configuration from: {config_file}")
    else:
        # Use command line parameters
        if not input_dir or not output_dir:
            raise typer.BadParameter(
                "Either --config-file or both --input-dir and --output-dir must be provided"
            )

        config = AnalysisConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            logs_subdir=logs_dir,
            monitoring_subdir=monitoring_dir,
        )

    analyzer = BenchmarkAnalyzer(config)

    # Process results
    typer.echo("Starting benchmark analysis...")
    analyzer.process_results()

    typer.echo(f"Analysis complete. Results in: {config.output_dir}")


if __name__ == "__main__":
    app()
