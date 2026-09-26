"""
Central configuration for the Rule Transformation Agent.

Everything the agent needs (API keys, model names, file paths, tunable
behaviour knobs) is read once, here, from environment variables / a local
.env file. Every other module imports `settings` from this file instead of
calling os.environ directly, so there is exactly one place to change when
you move from a laptop prototype to an AWS deployment.
"""
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# Load the .env file (if present) into the process environment.
load_dotenv()


def _get(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str
    generation_model: str
    classification_model: str
    embedding_model: str

    # Local data
    ground_truth_xlsx: Path
    chroma_persist_dir: Path

    # Agent behaviour
    max_repair_attempts: int
    retrieval_top_k: int

    # AWS (optional, only used by the production deployment path)
    aws_region: str
    aws_profile: str


def load_settings() -> Settings:
    return Settings(
        openai_api_key=_get("OPENAI_API_KEY", required=False),
        generation_model=_get("GENERATION_MODEL", "gpt-4.1"),
        classification_model=_get("CLASSIFICATION_MODEL", "gpt-4.1-mini"),
        embedding_model=_get("EMBEDDING_MODEL", "text-embedding-3-small"),
        ground_truth_xlsx=Path(_get(
            "GROUND_TRUTH_XLSX",
            "./data/Anonymised_Executable_Rule_Ground_Truth_Dataset.xlsx",
        )),
        chroma_persist_dir=Path(_get("CHROMA_PERSIST_DIR", "./data/chroma_store")),
        max_repair_attempts=int(_get("MAX_REPAIR_ATTEMPTS", "2")),
        retrieval_top_k=int(_get("RETRIEVAL_TOP_K", "5")),
        aws_region=_get("AWS_REGION", "eu-west-1"),
        aws_profile=_get("AWS_PROFILE", "default"),
    )


settings = load_settings()
