"""Command-line interface for benchmark analysis"""

from pathlib import Path
from typing import Optional

import typer

from ..core.analysis import BenchmarkAnalyzer
from ..schemas.benchmark import AnalysisConfig
from ..utils.paths import load_analysis_config_from_yaml

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
    config_file: Optional[Path] = typer.Option(
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

        config = AnalysisConfig(input_dir=input_dir, output_dir=output_dir)

    analyzer = BenchmarkAnalyzer(config)

    # Process results
    typer.echo("Starting benchmark analysis...")
    analyzer.process_results()

    typer.echo(f"Analysis complete. Results in: {config.output_dir}")


if __name__ == "__main__":
    app()
