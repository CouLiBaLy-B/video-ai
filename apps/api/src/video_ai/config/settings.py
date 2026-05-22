"""Application settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    storage_root: Path = Field(default=Path(".data/storage"))
    max_upload_bytes: int = 10 * 1024 * 1024
    vllm_text_base_url: str = "http://localhost:8000/v1"
    vllm_vision_base_url: str = "http://localhost:8001/v1"
    vllm_api_key: str = "not-needed-local"
    vllm_text_model: str = "Qwen/Qwen3.6-35B-A3B"
    vllm_vision_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vllm_timeout_seconds: float = 60.0


def get_settings() -> Settings:
    """Return settings instance."""
    return Settings()
