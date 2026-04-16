"""
DDR-AI-Builder Configuration Module.

Centralizes all application settings including API keys, model parameters,
file paths, and processing thresholds. Settings are loaded from environment
variables with sensible defaults for development.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "outputs"
SAMPLE_INPUT_DIR = BASE_DIR / "sample_inputs"
TEMPLATE_DIR = BASE_DIR / "generation" / "templates"
TEMP_IMAGE_DIR = BASE_DIR / "outputs" / "extracted_images"

# Ensure directories exist
for _dir in [OUTPUT_DIR, SAMPLE_INPUT_DIR, TEMPLATE_DIR, TEMP_IMAGE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# LLM Provider Settings
# ──────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "anthropic"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# LLM Parameters
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "2.0"))

# ──────────────────────────────────────────────
# Semantic Similarity Settings
# ──────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
CONFLICT_SIMILARITY_THRESHOLD = float(
    os.getenv("CONFLICT_SIMILARITY_THRESHOLD", "0.70")
)

# ──────────────────────────────────────────────
# PDF Parsing Settings
# ──────────────────────────────────────────────
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "100"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "100"))
MAX_IMAGE_SIZE_MB = float(os.getenv("MAX_IMAGE_SIZE_MB", "10.0"))
HEADING_FONT_SIZE_THRESHOLD = float(
    os.getenv("HEADING_FONT_SIZE_THRESHOLD", "12.0")
)

# ──────────────────────────────────────────────
# Severity Levels
# ──────────────────────────────────────────────
SEVERITY_LEVELS = {
    "critical": {
        "label": "Critical",
        "color": "#DC2626",
        "priority": 1,
        "description": "Immediate attention required; safety risk or major damage.",
    },
    "high": {
        "label": "High",
        "color": "#EA580C",
        "priority": 2,
        "description": "Significant issue; repair within 30 days.",
    },
    "medium": {
        "label": "Medium",
        "color": "#CA8A04",
        "priority": 3,
        "description": "Moderate issue; monitor and plan repairs.",
    },
    "low": {
        "label": "Low",
        "color": "#16A34A",
        "priority": 4,
        "description": "Minor issue; address during routine maintenance.",
    },
    "informational": {
        "label": "Informational",
        "color": "#2563EB",
        "priority": 5,
        "description": "No action needed; noted for reference.",
    },
}

# ──────────────────────────────────────────────
# Report Metadata Defaults
# ──────────────────────────────────────────────
DEFAULT_REPORT_TITLE = "Detailed Diagnostic Report (DDR)"
DEFAULT_COMPANY_NAME = os.getenv("COMPANY_NAME", "DDR-AI-Builder")
REPORT_VERSION = "1.0"


def get_llm_config() -> dict:
    """Return the active LLM configuration as a dictionary."""
    if LLM_PROVIDER == "anthropic":
        return {
            "provider": "anthropic",
            "api_key": ANTHROPIC_API_KEY,
            "model": ANTHROPIC_MODEL,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        }
    return {
        "provider": "openai",
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }


def validate_config() -> list[str]:
    """
    Validate essential configuration values.

    Returns:
        List of validation error messages. Empty list means all valid.
    """
    errors = []
    cfg = get_llm_config()
    if not cfg["api_key"]:
        errors.append(
            f"API key for '{cfg['provider']}' is not set. "
            f"Set {'OPENAI_API_KEY' if cfg['provider'] == 'openai' else 'ANTHROPIC_API_KEY'} "
            f"in your .env file."
        )
    return errors
