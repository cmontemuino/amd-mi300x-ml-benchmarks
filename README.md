# amd-mi300x-ml-benchmarks
Comprehensive machine learning benchmarking framework for AMD MI300X GPUs on Dell PowerEdge XE9680 hardware. Supports both inference (vLLM) and training workloads with containerized test suites, hardware monitoring, and analysis tools for performance, power efficiency, and scalability research across the complete ML pipeline.

## Quick Start with Sample Data

The repository includes sample benchmark results for immediate testing:

```shell
# Clone and setup
git clone https://github.com/cmontemuino/amd-mi300x-ml-benchmarks.git
cd amd-mi300x-ml-benchmarks
uv sync --extra analysis

# Run analysis on sample dataset
uv run analyze-results --input-dir datasets/sample-results --output-dir analysis/sample-output

# Or use the Python API
uv run python -c "
from amd_bench.schemas.examples import sample_dataset_example
analyzer = sample_dataset_example()
summary = analyzer.get_results_summary()
print(f'Processed {summary["total_results"]} results')
print(f'Models: {summary["models"]}')
"
```

## Analysis Commands

### Basic Analysis

```shell
# Analyze results with command line parameters

uv run analyze-results --input-dir datasets/sample-results --output-dir analysis/sample-output

# Using YAML configuration
uv run analyze-results run --config-file config/analysis-config.yaml
```

### Python API Usage

```python
from pathlib import Path
from amd_bench.core.analysis import BenchmarkAnalyzer
from amd_bench.schemas.benchmark import AnalysisConfig

# Basic configuration

config = AnalysisConfig(
    input_dir=Path("datasets/sample-results"),
    output_dir=Path("analysis/sample-output")
)

analyzer = BenchmarkAnalyzer(config)
analyzer.process_results()

# Get summary of results
summary = analyzer.get_results_summary()
print(f"Analyzed {summary['total_results']} benchmark results")
```

### Generated Output Structure

After running analysis, you'll find:

```text
analysis/sample-output/
├── plots/
│ ├── batch_size_scaling.png
│ ├── batch_size_scaling_by_memory.png
│ ├── latency_analysis.png
│ ├── memory_efficiency.png
│ └── throughput_comparison.png
├── reports/
│ ├── analysis_summary.json
│ └── benchmark_analysis_report.md
└── tables/
  ├── batch_size_analysis.csv
  ├── memory_utilization_analysis.csv
  └── model_performance_summary.csv
```

### Creating Custom Configurations

Create a YAML configuration file for custom analysis:

```shell
# config/custom-analysis.yaml

input_dir: "datasets/sample-results"
output_dir: "analysis/custom-output"
results_pattern: "*.json"
include_hardware_metrics: true
generate_plots: true

filename_formats:
    pattern: "([^]+)([^_]+)_bs(\d+)in(\d+)out(\d+)([^]+)mem([\d,\.]+)(.+)"
    groups:
    model: 1
    benchmark_type: 2
    batch_size: 3
    input_length: 4
    output_length: 5
    dtype: 6
    memory_util: 7
    timestamp: 8
    description: "Standard vLLM format"
    priority: 1

default_parameters:
benchmark_type: "latency"
memory_util: "0.8"
```

## 📊 Understanding Results

After running analysis, learn how to interpret your results:
- **[Complete Analysis Guide](docs/user-guide/analysis-guide.md)** - Comprehensive guide to understanding benchmark results

## 🔧 Development

**For Contributors:** See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development setup and guidelines.

**For Users:** The quick start above is sufficient for running analysis.

## Research Reproducibility

This project follows research software engineering best practices:

- **Reproducible environments**: Locked dependencies with `uv.lock`
- **Data validation**: Pydantic schemas for all data structures
- **Comprehensive logging**: Structured logs for all operations
- **Statistical rigor**: Proper statistical analysis methods
- **Configuration management**: YAML-based configuration system