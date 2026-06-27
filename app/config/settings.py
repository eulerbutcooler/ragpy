
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    cors_origins: list[str] = Field(default_factory=list, validation_alias="CORS_ORIGINS")

    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2:3b", validation_alias="OLLAMA_MODEL")

    llm_base_url: str = Field(default="http://localhost:8000/v1", validation_alias="LLM_BASE_URL")
    llm_condense_base_url: str = Field(default="http://localhost:8000/v1", validation_alias="LLM_CONDENSE_BASE_URL")
    llm_model: str = Field(default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", validation_alias="LLM_MODEL")
    llm_condense_model: str = Field(default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", validation_alias="LLM_CONDENSE_MODEL")
    llm_api_key: str = Field(default="fake-key", validation_alias="LLM_API_KEY")


    embed_model_name: str = Field(default="nomic-embed-text-v1.5", validation_alias="EMBED_MODEL_NAME")
    reranker_model_name: str = Field(default="BAAI/bge-reranker-base", validation_alias="RERANKER_MODEL_NAME")

    retrieval_top_k: int = Field(default=15, validation_alias="RETRIEVAL_TOP_K")
    reranker_top_n: int = Field(default=3, validation_alias="RERANKER_TOP_N")
    hybrid_search_enabled: bool = Field(default=True, validation_alias="HYBRID_SEARCH_ENABLED")
    rrf_k: int = Field(default=60, validation_alias="RRF_K")

    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="course_documents", validation_alias="QDRANT_COLLECTION_NAME")
    qdrant_vector_size: int = Field(default=768, validation_alias="QDRANT_VECTOR_SIZE")

    minio_endpoint: str = Field(default="http://localhost:9000", validation_alias="MINIO_ENDPOINT")
    minio_region: str = Field(default="us-east-1", validation_alias="MINIO_REGION")
    minio_access_key_id: str = Field(default="minioadmin", validation_alias="MINIO_ACCESS_KEY_ID")
    minio_secret_access_key: str = Field(default="minioadmin", validation_alias="MINIO_SECRET_ACCESS_KEY")
    minio_use_ssl: bool = Field(default=False, validation_alias="MINIO_USE_SSL")

    nats_url: str = Field(default="nats://localhost:4222", validation_alias="NATS_URL")
    nats_user: str | None = Field(default=None, validation_alias="NATS_USER")
    nats_password: str | None = Field(default=None, validation_alias="NATS_PASSWORD")

    internal_token: str = Field(default="development_secret_token", validation_alias="INTERNAL_TOKEN")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            # If user provides JSON array, let Pydantic handle it.
            if s.startswith("[") and s.endswith("]"):
                return value  # type: ignore[return-value]
            return [part.strip() for part in s.split(",") if part.strip()]
        return [str(value).strip()] if str(value).strip() else []


settings = Settings()

