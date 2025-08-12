"""Usage examples for the benchmark analysis features"""

from pathlib import Path
from typing import List, Tuple

from ..core.analysis import BenchmarkAnalyzer
from ..schemas.benchmark import AnalysisConfig


def get_sample_dataset_path() -> Path:
    """Get the path to the sample dataset directory"""
    # Get the project root (assuming examples.py is in src/amd_bench/schemas/)
    project_root = Path(__file__).parent.parent.parent.parent
    return project_root / "datasets" / "sample-results"


def basic_usage_example() -> BenchmarkAnalyzer:
    """Basic usage with default configuration using sample dataset"""
    sample_path = get_sample_dataset_path()

    config = AnalysisConfig(input_dir=sample_path, output_dir=Path("analysis/sample-output"))
    analyzer = BenchmarkAnalyzer(config)

    # Process sample results
    # analyzer.process_results()
    return analyzer


def sample_dataset_example() -> BenchmarkAnalyzer:
    """Complete example using the included sample dataset"""
    sample_path = get_sample_dataset_path()

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at {sample_path}",
            "Please ensure the datasets/ directory is present in the project root",
        ) from None

    config = AnalysisConfig(
        input_dir=sample_path,
        output_dir=Path("analysis/sample-analysis"),
        filename_formats=[
            {
                "pattern": r"([^_]+)_([^_]+)_bs(\d+)_in(\d+)_out(\d+)_([^_]+)_mem([\d,\.]+)_(.+)",
                "groups": {
                    "model": 1,
                    "benchmark_type": 2,
                    "batch_size": 3,
                    "input_length": 4,
                    "output_length": 5,
                    "dtype": 6,
                    "memory_util": 7,
                    "timestamp": 8,
                },
                "description": "Standard vLLM format",
                "priority": 1,
            }
        ],
        default_parameters={
            "benchmark_type": "latency",
            "memory_util": "0.8",
        },
    )

    analyzer = BenchmarkAnalyzer(config)

    print("Sample dataset contains:")
    json_files = list(config.input_dir.glob("*.json"))
    for file in json_files[:3]:  # Show first 3 files
        print(f"  - {file.name}")
    if len(json_files) > 3:
        print(f"  ... and {len(json_files) - 3} more files")

    return analyzer


def custom_format_example() -> BenchmarkAnalyzer:
    """Custom filename format configuration with sample data"""
    sample_path = get_sample_dataset_path()

    config = AnalysisConfig(
        input_dir=sample_path,
        output_dir=Path("analysis/custom-analysis"),
        filename_formats=[
            {
                "pattern": r"([^_]+)_([^_]+)_bs(\d+)_in(\d+)_out(\d+)_([^_]+)_mem([\d,\.]+)_(.+)",
                "groups": {
                    "model": 1,
                    "benchmark_type": 2,
                    "batch_size": 3,
                    "input_length": 4,
                    "output_length": 5,
                    "dtype": 6,
                    "memory_util": 7,
                    "timestamp": 8,
                },
                "description": "Standard vLLM format",
                "priority": 1,
            },
            {
                "pattern": r"mi300x_([^_]+)_perf_batch(\d+)_(.+)",
                "groups": {"model": 1, "batch_size": 2, "timestamp": 3},
                "description": "MI300X performance format",
                "priority": 2,
            },
        ],
        default_parameters={
            "benchmark_type": "throughput",
            "memory_util": "0.85",
        },
    )

    analyzer = BenchmarkAnalyzer(config)
    return analyzer


def yaml_config_example() -> BenchmarkAnalyzer:
    """Loading configuration from YAML file with sample dataset"""
    import yaml

    # Create sample config if it doesn't exist
    sample_config_path = get_sample_dataset_path().parent / "configs" / "sample-analysis.yaml"

    if not sample_config_path.exists():
        create_sample_config_file(sample_config_path)

    # Load from YAML file
    with open(sample_config_path) as f:
        yaml_config = yaml.safe_load(f)

    config = AnalysisConfig(**yaml_config)
    analyzer = BenchmarkAnalyzer(config)
    return analyzer


def create_sample_config_file(config_path: Path) -> None:
    """Create a sample YAML configuration file"""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    sample_config = {
        "input_dir": str(get_sample_dataset_path()),
        "output_dir": "analysis/yaml-sample-output",
        "results_pattern": "*.json",
        "include_hardware_metrics": True,
        "generate_plots": True,
        "filename_formats": [
            {
                "pattern": r"([^_]+)_([^_]+)_bs(\d+)_in(\d+)_out(\d+)_([^_]+)_mem([\d,\.]+)_(.+)",
                "groups": {
                    "model": 1,
                    "benchmark_type": 2,
                    "batch_size": 3,
                    "input_length": 4,
                    "output_length": 5,
                    "dtype": 6,
                    "memory_util": 7,
                    "timestamp": 8,
                },
                "description": "Standard vLLM format",
                "priority": 1,
            }
        ],
        "default_parameters": {
            "benchmark_type": "latency",
            "memory_util": "0.8",
            "batch_size": "1",
            "input_length": "128",
            "output_length": "128",
            "dtype": "float16",
        },
        "parameter_sanitizers": {
            "memory_util": "decimal_separator",
            "batch_size": "numeric_only",
            "input_length": "numeric_only",
            "output_length": "numeric_only",
        },
    }

    import yaml

    with open(config_path, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)


# Validation functions
def validate_sample_dataset() -> Tuple[bool, str]:
    """Validate that the sample dataset is properly structured"""
    sample_path = get_sample_dataset_path()

    if not sample_path.exists():
        return False, f"Sample dataset directory not found: {sample_path}"

    json_files = list(sample_path.glob("*.json"))
    if len(json_files) == 0:
        return False, f"No JSON files found in {sample_path}"

    return True, f"Found {len(json_files)} sample result files"


def list_sample_files() -> List[Path]:
    """List all files in the sample dataset"""
    sample_path = get_sample_dataset_path()

    if not sample_path.exists():
        print(f"Sample dataset not found at: {sample_path}")
        return []

    json_files = sorted(sample_path.glob("*.json"))
    print(f"Sample dataset contains {len(json_files)} files:")
    for file in json_files:
        print(f"  - {file.name}")

    return json_files
