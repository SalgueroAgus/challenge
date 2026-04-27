from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "development"
    app_version: str = "0.1.0"

    # LLM provider — "ollama" (local) or "groq" (cloud free tier)
    llm_provider: str = "ollama"

    # Ollama (used when LLM_PROVIDER=ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"

    # Groq (used when LLM_PROVIDER=groq)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "documents"

    # Embeddings
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # RAG
    rag_top_k: int = 8
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200

    # Data directories (relative to backend/ working directory)
    data_dir: str = "../data"

    # CORS — space-separated list of allowed frontend origins
    cors_origins: str = "http://localhost:5173 http://localhost:5174 http://localhost"

    # LangFuse observability (optional — tracing disabled if keys are empty)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Auth
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 h
    auth_username: str = "admin"
    # bcrypt hash of "changeme" — override via AUTH_PASSWORD_HASH in .env
    auth_password_hash: str = "$2b$12$IO/CRwLA7Lg2.PfR47Drgu2XGNhtKsR1zm0RYaG4sJoDwSzxhA6Zi"


settings = Settings()
