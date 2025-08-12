# Sample Dataset for AMD MI300X Benchmarks

This directory contains sample benchmark results for testing and demonstration purposes.

## Contents

### `sample-results/containerized/`

Contains benchmark results from vLLM inference tests on AMD MI300X hardware:

- **Models tested**: DialoGPT-medium, Llama-3.1-8B-Instruct
- **Benchmark types**: Latency optimization
- **Configurations**: Various batch sizes, memory utilizations
- **Hardware**: Dell PowerEdge XE9680 with AMD MI300X GPUs

### File Naming Convention

Files follow the pattern:

```text
{model}_{benchmark_type}_bs{batch_size}_in{input_length}out{output_length}{dtype}mem{memory_util}{timestamp}.json
```

Examples:
- `Llama-3.1-8B_latency_bs1_in128_out128_float16_mem0.8_20250807_185823.json`
- `DialoGPT-medium_latency_bs1_in128_out128_float16_mem0.8_20250807_184434.json`

### Sample Data Statistics

- **Total files**: 13 JSON files
- **Date range**: August 7, 2025
- **Batch sizes**: 1, 8, 32
- **Input/Output lengths**: 128, 1024 tokens
- **Memory utilizations**: 0.8, 0.9
- **Data type**: float16

## Usage

### Quick Start

```python
from amd_bench.schemas.examples import sample_dataset_example

# Run analysis on sample data
analyzer = sample_dataset_example()
```

### Command Line

#### Analyze sample dataset

```shell
analyze-results run --input-dir datasets/sample-results --output-dir analysis/sample-output
```

#### Use YAML configuration

```shell
analyze-results run --config-file datasets/configs/sample-analysis.yaml
```

## Complete Dataset

This is a small subset for testing. The complete research dataset is available at:
[**https://github.com/cmontemuino/amd-mi300x-research-data**](https://github.com/cmontemuino/amd-mi300x-research-data)

The complete dataset includes:
- Additional models (Llama-3.1-70B, Mistral, Qwen)
- Throughput benchmarks
- Hardware monitoring data
- Power consumption metrics
- Multi-GPU configurations

## Data Schema

Each JSON file contains:

```json
{
    "avg_latency": 0.7171628322103061,
    "latencies": [0.717713778023608, ...],
    "percentiles": {
        "10": 0.716384768707212,
        "25": 0.7165017670486122,
        "50": 0.7168599735596217,
        "75": 0.7176406020007562,
        "90": 0.7181383826304227,
        "99": 0.7188986453670076
    }
}
```
