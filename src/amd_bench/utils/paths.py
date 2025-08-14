"""Path management utilities"""

from pathlib import Path
from typing import Union

import yaml

from ..schemas.benchmark import AnalysisConfig


def ensure_directory(path: Union[Path, str]) -> Path:
    """Ensure directory exists, create it otherwise"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(path: Union[Path, str], description: str = "File") -> Path:
    """Validate file exists"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}") from None
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {path}") from None
    return path


def load_analysis_config_from_yaml(config_path: Union[Path, str]) -> AnalysisConfig:
    """Load AnalysisConfig from YAML file"""
    config_path = Path(config_path)
    validate_file_exists(config_path, "Analysis configuration file")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        if yaml_data is None:
            raise ValueError(f"YAML configuration file is empty: {config_path}") from None

        # Convert string paths to Path objects
        if "input_dir" in yaml_data:
            yaml_data["input_dir"] = Path(yaml_data["input_dir"])
        if "output_dir" in yaml_data:
            yaml_data["output_dir"] = Path(yaml_data["output_dir"])
        if "logs_subdir" in yaml_data:
            yaml_data["logs_subdir"] = str(yaml_data["logs_subdir"])
        if "monitoring_subdir" in yaml_data:
            yaml_data["monitoring_subdir"] = str(yaml_data["monitoring_subdir"])

        return AnalysisConfig(**yaml_data)

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {config_path}: {e}") from e
