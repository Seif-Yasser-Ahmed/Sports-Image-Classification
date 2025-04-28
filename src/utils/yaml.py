import yaml
import importlib.resources
from functools import lru_cache


class Config:
    """
    Load and cache the params.yaml bundled alongside this module.
    """
    @staticmethod
    @lru_cache(maxsize=1)
    def load() -> dict:
        # __package__ is "pipeline.utils" here
        with importlib.resources.open_text(__package__, "params.yaml") as f:
            return yaml.safe_load(f)
