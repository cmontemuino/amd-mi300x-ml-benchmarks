# Sample AMD MI300X Benchmark Data

This directory contains representative samples of AMD MI300X benchmark data demonstrating the complete experimental pipeline.

## Data Structure

```text
sample-results/
├── containerized/ # JSON benchmark results (4 experiments)
├── logs/ # Execution logs (5 files)
└── monitoring/ # Hardware monitoring CSV files (20 files)
```

## Experimental Design

**Model**: Llama-3.1-8B (representative 8B parameter model)  
**Benchmark Type**: Latency-focused inference  
**Parameter Variations**:
- **Batch Size**: 1 (latency-optimized) vs 8 (throughput-focused)
- **Memory Utilization**: 0.8 vs 0.9 (resource efficiency study)
- **Data Type**: float16 (production standard)

## Hardware Context

- **Platform**: Dell PowerEdge XE9680 with 8× AMD MI300X GPUs
- **Container**: vLLM inference framework
- **Monitoring**: Comprehensive system metrics (CPU, GPU power/temp/usage, memory)

### File Naming Convention

Files follow the pattern:

```text
{model}_{benchmark_type}_bs{batch_size}_in{input_length}out{output_length}{dtype}mem{memory_util}{timestamp}.json
```

Examples:
- `Llama-3.1-8B_latency_bs1_in128_out128_float16_mem0.8_20250807_185823.json`

## Complete Dataset

This is a small subset for testing. The complete research dataset is available at:

👉 [**https://github.com/cmontemuino/amd-mi300x-research-data**](https://github.com/cmontemuino/amd-mi300x-research-data)

The complete dataset includes:
- Additional models (Llama-3.1-70B, Mistral, Qwen)
- Throughput benchmarks
- Hardware monitoring data
- Power consumption metrics
- Multi-GPU configurations


## Usage Example

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
