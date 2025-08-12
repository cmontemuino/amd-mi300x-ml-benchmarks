# Contributing to AMD MI300X ML Benchmarks

## Development Setup

### Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

#### 1. Clone and Setup

```shell
# Clone the repository
git clone https://github.com/cmontemuino/amd-mi300x-ml-benchmarks.git
cd amd-mi300x-ml-benchmarks

# Create virtual environment and install dependencies
uv venv --python 3.12

# Install runtime dependencies
uv sync

# Install development dependencies
uv sync --extra dev

# Install analysis dependencies
uv sync --extra analysis

# Install both development ana analysis dependencies

uv sync --extra analysis development

# Install all dependencies
uv sync --all-extras
```

#### 2. Activate Environment

```shell
# Activate the virtual environment
source .venv/bin/activate

# Verify tooling
ruff check .
black --check .
mypy src
pytest -q

# Verify installation
analyze-results --help 
```

#### 3. Development Workflow

```shell
# Unit tests only
uv run pytest tests/unit/ -v

# Integration tests only  
uv run pytest tests/integration/ -v

# All tests
uv run pytest tests/ -v

# Run type checking
uv run mypy src/

# Check format
uv run black --check src/ tests/

# See what black wants to reformat, if any
uv run black --diff --color src/ tests/

# Lint code with Ruff
uv run ruff check src/ tests/

# Manually reformat or just go ahead with suggestions
uv run black src/ tests

# Run analysis on sample data
uv run analyze-results --input-dir datasets/sample-results --output-dir analysis/sample-output
```

### Testing the Analysis Pipeline

```shell
# Test with sample data

uv run python -c "
from amd_bench.schemas.examples import sample_dataset_example
analyzer = sample_dataset_example()
analyzer.process_results()
print('✅ Analysis pipeline working correctly!')
"

# Validate sample dataset
uv run python -c "
from amd_bench.schemas.examples import validate_sample_dataset, list_sample_files
valid, msg = validate_sample_dataset()
print(f'Dataset validation: {msg}')
list_sample_files()
"
```

### Project Structure

```text
src/amd_bench/ # Main package
├── cli/ # Command-line interfaces
├── core/ # Business logic
├── utils/ # Utility functions
└── schemas/ # Data validation
```

### Key Commands

```shell
# Analysis workflow
uv run analyze-results --help
uv run analyze-results --input results/ --output analysis/
```

## Testing Guidelines

### Basic Test Execution

```shell
# Run all tests for the analysis module
uv run pytest tests/unit/core/test_analysis.py -v

# Run only the static method tests
uv run pytest tests/unit/core/test_analysis.py::TestBenchmarkAnalyzerStaticMethods -v

# Run a specific test class
uv run pytest tests/unit/core/test_analysis.py::TestBenchmarkAnalyzerStaticMethods::TestSanitizeParameterValue -v

# Run with coverage
uv run pytest tests/unit/core/test_analysis.py --cov=src/amd_bench.core.analysis
```

### Advanced Test Execution

```shell
# Run tests with detailed output
uv run pytest tests/unit -v -s

# Run tests with coverage report
uv run pytest tests/unit --cov-report=html --cov=src/amd_bench

# Run tests and show which ones are slowest
uv run pytest tests/unit --durations=10

# Run only fast tests
uv run pytest -m "not slow"
```

## Documentation

```shell
# Build documentation
uv run mkdocs build

# Serve documentation locally
uv run mkdocs server
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
    - Ensure virtual environment is activated: `source .venv/bin/activate`
    - Reinstall dependencies: `uv sync --extra analysis`

2. **Command Line Syntax**:
    - Use single line commands or proper line continuation with `\`
    - Ensure proper quoting of arguments

3. **File Not Found**:
    - Verify sample dataset exists: `ls datasets/sample-results/`
    - Use absolute paths if needed

### Environment Setup

```shell
# Verify environment
uv run python -c "import amd_bench; print('✅ Package imported successfully')"

# Check sample data
ls datasets/sample-results/*.json | head -5
```