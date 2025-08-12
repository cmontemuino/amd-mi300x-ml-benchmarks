# Quick Start Guide

Get up and running with AMD MI300X benchmarks in under 5 minutes.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Git

## Installation

```shell
# Clone and setup
git clone https://github.com/cmontemuino/amd-mi300x-ml-benchmarks.git
cd amd-mi300x-ml-benchmarks
uv sync --extra analysis
```

## Run Analysis

```shell
# Run analysis on sample dataset
uv run analyze-results --input-dir datasets/sample-results --output-dir analysis/sample-output
```

## What's Next?

- [📖 Read the Analysis Guide](../user-guide/analysis-guide.md)
- [⚙️ Configure Custom Analysis](../user-guide/configuration.md)
- [🔧 Contribute to the Project](../CONTRIBUTING.md)