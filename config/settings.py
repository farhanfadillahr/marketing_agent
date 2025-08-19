import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings configuration."""
    
    # Environment
    environment: str = "development"
    
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    
    # GEMINI api key
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"

    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "embeddings_example"
    qdrant_marketing_collection: str = "marketing_embeddings"
    qdrant_is_https: bool = False
    
    # Application
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",)

# Global settings instance
settings = Settings()