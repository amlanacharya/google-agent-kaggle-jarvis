"""Configuration management for Jarvis Assistant."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = Field(default="Jarvis Assistant", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    env: str = Field(default="development", env="ENV")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")

    # Google Cloud & Gemini
    google_api_key: str = Field(env="GOOGLE_API_KEY")
    google_cloud_project: Optional[str] = Field(default=None, env="GOOGLE_CLOUD_PROJECT")
    google_application_credentials: Optional[str] = Field(default=None, env="GOOGLE_APPLICATION_CREDENTIALS")
    gemini_model: str = Field(default="gemini-pro", env="GEMINI_MODEL")

    # OpenAI (Fallback)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")

    # Anthropic Claude (Fallback)
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-3-sonnet-20240229", env="CLAUDE_MODEL")

    # Vector Database
    chromadb_host: str = Field(default="localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(default=8000, env="CHROMADB_PORT")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_env: Optional[str] = Field(default=None, env="PINECONE_ENV")
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")

    # Redis Cache
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")

    # MongoDB
    mongodb_uri: str = Field(default="mongodb://localhost:27017", env="MONGODB_URI")
    mongodb_db: str = Field(default="jarvis_db", env="MONGODB_DB")

    # Message Queue
    rabbitmq_url: str = Field(default="amqp://guest:guest@localhost:5672", env="RABBITMQ_URL")
    kafka_bootstrap_servers: str = Field(default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")

    # Google Calendar API
    google_calendar_credentials: Optional[str] = Field(default=None, env="GOOGLE_CALENDAR_CREDENTIALS")

    # Gmail API
    gmail_credentials: Optional[str] = Field(default=None, env="GMAIL_CREDENTIALS")

    # Speech Services
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: Optional[str] = Field(default=None, env="ELEVENLABS_VOICE_ID")

    # Web Search
    serper_api_key: Optional[str] = Field(default=None, env="SERPER_API_KEY")
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")

    # IoT Integration
    home_assistant_url: Optional[str] = Field(default=None, env="HOME_ASSISTANT_URL")
    home_assistant_token: Optional[str] = Field(default=None, env="HOME_ASSISTANT_TOKEN")
    mqtt_broker: str = Field(default="localhost", env="MQTT_BROKER")
    mqtt_port: int = Field(default=1883, env="MQTT_PORT")
    mqtt_username: str = Field(default="", env="MQTT_USERNAME")
    mqtt_password: str = Field(default="", env="MQTT_PASSWORD")

    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")

    # Security
    secret_key: str = Field(env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=60, env="JWT_EXPIRATION_MINUTES")

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")

    # Memory Settings
    max_context_messages: int = Field(default=10, env="MAX_CONTEXT_MESSAGES")
    max_memory_vectors: int = Field(default=10000, env="MAX_MEMORY_VECTORS")
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")

    # Task Processing
    max_concurrent_tasks: int = Field(default=5, env="MAX_CONCURRENT_TASKS")
    task_timeout_seconds: int = Field(default=300, env="TASK_TIMEOUT_SECONDS")

    # Voice Settings
    wake_word: str = Field(default="jarvis", env="WAKE_WORD")
    voice_language: str = Field(default="en-US", env="VOICE_LANGUAGE")
    speech_recognition_timeout: int = Field(default=5, env="SPEECH_RECOGNITION_TIMEOUT")

    # Development
    reload_on_change: bool = Field(default=True, env="RELOAD_ON_CHANGE")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8000", env="CORS_ORIGINS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def chromadb_url(self) -> str:
        """Construct ChromaDB URL."""
        return f"http://{self.chromadb_host}:{self.chromadb_port}"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
