"""Pydantic models for benchmark data validation"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkMetrics(BaseModel):
    """Core benchmark performance metrics"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    avg_latency: float = Field(..., gt=0, description="Average latency in seconds")
    latency_std: float = Field(0.0, ge=0, description="Standard deviation of latencies")
    p50_latency: float = Field(0.0, ge=0, description="50th percentile latency")
    p90_latency: float = Field(0.0, ge=0, description="90th percentile latency")
    p95_latency: float = Field(0.0, ge=0, description="95th percentile latency")
    p99_latency: float = Field(0.0, ge=0, description="99th percentile latency")
    throughput: float = Field(0.0, ge=0, description="Throughput in requests/second")
    tokens_per_second: float = Field(0.0, ge=0, description="Token generation rate")
    total_iterations: int = Field(..., gt=0, description="Number of benchmark iterations")


@dataclass
class FilenameFormat:
    """Configuration for filename parsing patterns"""

    pattern: str
    groups: Dict[str, int]  # Maps field names to regex group indices
    description: str
    priority: int = 100  # Lower numbers = higher priority


class ExperimentConfig(BaseModel):
    """Experiment configuration parameters"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    model: str = Field(
        ...,
        min_length=1,
        description="Model identifier",
        examples=["meta-llama/Llama-3.1-8B-Instruct", "microsoft/DialoGPT-medium"],
    )
    benchmark_type: Literal["latency", "throughput"] = Field(
        ..., description="Benchmark type", examples=["latency", "throughput"]
    )
    batch_size: int = Field(..., gt=0, description="Batch size for inference", examples=[1, 8, 32])
    input_length: int = Field(..., gt=0, description="Input sequence length", examples=[128, 512])
    output_length: int = Field(
        ..., gt=0, description="Output sequence length", examples=[128, 1024]
    )
    dtype: Literal["float16", "bfloat16", "float8", "int4"] = Field(
        ..., description="Data type precision", examples=["float16"]
    )
    memory_util: float = Field(
        ..., gt=0, le=1, description="GPU memory utilization fraction", examples=[0.8, 0.9, 0.95]
    )
    timestamp: str = Field(..., description="Experiment timestamp")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """
        Validate timestamp format.
        File names with results include human-readable strings like "20250812_143022", for example:
        "Llama-3.1-8B_latency_bs32_in128_out256_float16_mem0.9_20250812_143022.json"
        A custom validation that handles multiple timestamp formats from different sources is critical.
        """
        try:
            datetime.strptime(v, "%Y%m%d_%H%M%S")
            return v
        except ValueError:
            # Try alternatives
            for fmt in ["%Y-%m-%d_%H-%M-%S", "%Y%m%d%H%M%S"]:
                try:
                    datetime.strptime(v, fmt)
                    return v
                except ValueError:
                    continue
            raise ValueError(f"Invalid timestamp format: {v}") from None


class BenchmarkResult(BaseModel):
    """Complete benchmark result with metadata and metrics"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    # Metadata
    file: str = Field(..., description="Result filename")
    experiment_id: str = Field(..., description="Unique experiment identifier")

    # Configuration
    config: ExperimentConfig

    # Performance metrics
    metrics: BenchmarkMetrics

    @property
    def model_short_name(self) -> str:
        """Extract short model name for display"""
        return self.config.model.split("/")[-1].replace("-Instruct", "")

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score [throughput/latency]"""
        if self.metrics.avg_latency > 0:
            return self.metrics.throughput / self.metrics.avg_latency
        return 0.0


class HardwareMetrics(BaseModel):
    """Hardware monitoring metrics"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    gpu_usage_percent: Optional[float] = Field(
        default=None, ge=0, le=100, description="GPU usage in percentage"
    )
    gpu_power_watts: Optional[float] = Field(
        default=None, ge=0, description="GPU power consumption in Watts"
    )
    gpu_temp_celsius: Optional[float] = Field(
        default=None, ge=0, description="GPU temperature in Celsius"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None, ge=0, le=100, description="CPU usage in percentage"
    )
    memory_usage_percent: Optional[float] = Field(
        default=None, ge=0, le=100, description="Memory usage in percentage"
    )
    # This is live system monitoring data used for time-series analysis and correlation.
    # Automatic validation through Pydantic's built-in datetime parsing is enough.
    timestamp: datetime


class AnalysisConfig(BaseModel):
    """Configuration for analysis operations"""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    input_dir: Path = Field(
        ...,
        description="Input directory containing results",
        examples=[Path("datasets/sample-results")],
    )
    output_dir: Path = Field(
        ...,
        description="Output directory for data analysis",
        examples=[Path("analysis/sample-analysis")],
    )
    results_pattern: str = Field(
        default="*.json",
        description="Pattern to match result files",
        examples=["*_latency_*.json", "*_throughput_*.json"],
    )
    include_hardware_metrics: bool = Field(default=True, description="Include hardware analysis")
    generate_plots: bool = Field(default=True, description="Generate visualization plots")

    filename_formats: List[Dict[str, Any]] = Field(
        default=[
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
        description="Filename parsing configurations (ordered by priority)",
    )

    default_parameters: Dict[str, str] = Field(
        default={
            "model": "unknown",
            "benchmark_type": "inference",
            "batch_size": "1",
            "input_length": "512",
            "output_length": "128",
            "dtype": "float16",
            "memory_util": "0.0",
            "timestamp": "unknown",
        },
        description="Default parameter values for unparsed fields",
    )

    parameter_sanitizers: Dict[str, str] = Field(
        default={
            "memory_util": "decimal_separator",  # Replace , with .
            "batch_size": "numeric_only",  # Remove non-digits
            "input_length": "numeric_only",
            "output_length": "numeric_only",
        },
        description="Sanitization rules for extracted parameters",
    )

    @field_validator("filename_formats")
    @classmethod
    def validate_filename_formats(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate filename format configurations"""
        for i, fmt in enumerate(v):
            if "pattern" not in fmt or "groups" not in fmt:
                raise ValueError(f"Format {i}: must have 'pattern' and 'groups'")

            # Test if pattern compiles
            try:
                re.compile(fmt["pattern"])
            except re.error as e:
                raise ValueError(f"Format {i}: invalid regex pattern: {e}") from e

        return v

    @field_validator("input_dir")
    @classmethod
    def validate_input_dir(cls, v: Path) -> Path:
        """Validate input directory exists and is readable."""
        resolved = v.resolve()

        if not resolved.exists():
            raise ValueError(f"Input directory does not exist: {resolved}")

        if not resolved.is_dir():
            raise ValueError(f"Input path is not a directory: {resolved}")

        return resolved

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure output directory is resolved."""
        resolved = v.resolve()

        # Create parent directories if they don't exist
        resolved.parent.mkdir(parents=True, exist_ok=True)

        return resolved
