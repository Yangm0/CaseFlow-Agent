from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CASEFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    mode: Literal["offline_test", "live"] = "offline_test"
    database_url: str = Field(
        default="postgresql+psycopg://caseflow_app:caseflow_app_dev@127.0.0.1:5432/caseflow",
        validation_alias="DATABASE_URL",
    )
    seed_database_url: str | None = Field(default=None, validation_alias="SEED_DATABASE_URL")
    test_database_url: str | None = Field(default=None, validation_alias="TEST_DATABASE_URL")
    test_seed_database_url: str | None = Field(
        default=None, validation_alias="TEST_SEED_DATABASE_URL"
    )
    checkpoint_database_url: str | None = Field(
        default=None, validation_alias="CHECKPOINT_DATABASE_URL"
    )
    test_checkpoint_database_url: str | None = Field(
        default=None, validation_alias="TEST_CHECKPOINT_DATABASE_URL"
    )
    action_executor_database_url: str | None = Field(
        default=None, validation_alias="ACTION_EXECUTOR_DATABASE_URL"
    )
    mcp_server_url: str = Field(
        default="http://127.0.0.1:8001/mcp", validation_alias="CASEFLOW_MCP_SERVER_URL"
    )
    mcp_host: str = Field(default="127.0.0.1", validation_alias="CASEFLOW_MCP_HOST")
    mcp_port: int = Field(
        default=8001, gt=0, le=65535, validation_alias="CASEFLOW_MCP_PORT"
    )
    mcp_tool_timeout_seconds: float = Field(
        default=8, gt=0, le=60, validation_alias="CASEFLOW_MCP_TOOL_TIMEOUT_SECONDS"
    )
    embedding_api_key: SecretStr | None = Field(
        default=None, validation_alias="EMBEDDING_API_KEY"
    )
    embedding_provider: Literal["openai_compatible", "fastembed"] = Field(
        default="openai_compatible", validation_alias="EMBEDDING_PROVIDER"
    )
    embedding_base_url: str | None = Field(
        default=None, validation_alias="EMBEDDING_BASE_URL"
    )
    embedding_model: str | None = Field(default=None, validation_alias="EMBEDDING_MODEL")
    embedding_dimensions: int | None = Field(
        default=None, gt=0, le=2000, validation_alias="EMBEDDING_DIMENSIONS"
    )
    embedding_cache_dir: str = Field(
        default=".tmp/fastembed_cache", validation_alias="EMBEDDING_CACHE_DIR"
    )
    embedding_index_database_url: str | None = Field(
        default=None, validation_alias="EMBEDDING_INDEX_DATABASE_URL"
    )
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_MODEL")
    structured_output_method: Literal["json_schema", "function_calling"] = "json_schema"
    synthesis_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = (
        "none"
    )
    decision_timeout_seconds: float = Field(default=10, gt=0, le=60)
    synthesis_timeout_seconds: float = Field(default=10, gt=0, le=60)
    decision_max_tokens: int = Field(default=900, ge=64, le=4096)
    synthesis_max_tokens: int = Field(default=2048, ge=64, le=4096)
    max_model_calls: int = Field(default=10, ge=1, le=20)
    max_tool_calls: int = Field(default=8, ge=1, le=20)
    max_agent_seconds: float = Field(default=25, gt=0, le=120)
    sql_statement_timeout_ms: int = Field(default=3000, ge=100, le=30000)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
