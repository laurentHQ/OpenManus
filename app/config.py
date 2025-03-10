import threading
import tomllib
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


class GlobalSettings(BaseModel):
    """Global application settings."""
    default_llm: str = Field("default", description="Default LLM configuration to use")
    log_level: str = Field("INFO", description="Logging level")
    workspace_dir: str = Field("workspace", description="Workspace directory")
    max_retries: int = Field(3, description="Maximum number of retries for operations")


class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    temperature: float = Field(1.0, description="Sampling temperature")
    tokens_per_minute: int = Field(40000, description="Maximum tokens per minute rate limit")
    requests_per_minute: int = Field(60, description="Maximum requests per minute")


class AppConfig(BaseModel):
    global_: GlobalSettings = Field(alias="global")
    llm: Dict[str, LLMSettings]


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        
        # Load global settings
        global_settings = raw_config.get("global", {})
        
        # Load LLM settings
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "temperature": base_llm.get("temperature", 1.0),
            "tokens_per_minute": base_llm.get("tokens_per_minute", 40000),
            "requests_per_minute": base_llm.get("requests_per_minute", 60),
        }

        config_dict = {
            "global": global_settings,
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            }
        }

        self._config = AppConfig(**config_dict)

    @property
    def global_(self) -> GlobalSettings:
        """Get global settings."""
        return self._config.global_

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        """Get LLM settings."""
        return self._config.llm


config = Config()
