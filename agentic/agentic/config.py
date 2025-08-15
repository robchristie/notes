from __future__ import annotations
import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # OpenRouter
    openrouter_api_key: str = Field(alias="OPENROUTER_API_KEY")
    openrouter_model_fast: str = Field("qwen/qwen2.5-coder:latest", alias="OPENROUTER_MODEL_FAST")
    openrouter_model_reasoning: str = Field("anthropic/claude-3.7-sonnet", alias="OPENROUTER_MODEL_REASONING")
    openrouter_http_referrer: str | None = Field(None, alias="OPENROUTER_HTTP_REFERRER")
    openrouter_x_title: str | None = Field(None, alias="OPENROUTER_X_TITLE")

    # Brave
    brave_api_key: str = Field(alias="BRAVE_API_KEY")
    brave_default_country: str = Field("AU", alias="BRAVE_DEFAULT_COUNTRY")

    # Langfuse
    langfuse_host: str = Field(alias="LANGFUSE_HOST")
    langfuse_public_key: str = Field(alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(alias="LANGFUSE_SECRET_KEY")

    # Temporal
    temporal_address: str = Field("localhost:7233", alias="TEMPORAL_ADDRESS")
    temporal_namespace: str = Field("default", alias="TEMPORAL_NAMESPACE")

    # Misc
    dry_run: int = Field(0, alias="AGENTIC_DRY_RUN")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# -------- Langfuse tracing --------
from contextlib import contextmanager
import time
try:
    from langfuse import Langfuse
    _LF = Langfuse(
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        enabled=bool(settings.langfuse_public_key and settings.langfuse_secret_key),
    )
except Exception:
    _LF = None

@contextmanager
def span(name: str, **attrs):
    start = time.time()
    trace = None
    try:
        if _LF:
            trace = _LF.trace(name=name, input=attrs or {})
        yield trace
    except Exception as e:
        if trace:
            trace.update(output={"error": str(e)})
        raise
    finally:
        dur = time.time() - start
        if trace:
            trace.update(output={**(attrs or {}), "duration_s": round(dur, 4)})
