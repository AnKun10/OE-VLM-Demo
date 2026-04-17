"""Constants and defaults for the Qwen vLLM provider.

This is the ONLY file in this package that should be edited to tune
hyperparameters for Qwen-family models served via vLLM.
"""

# Regex patterns stripped from user text (quirk i — image-token leakage).
IMAGE_TOKEN_PATTERNS: tuple[str, ...] = (
    r"<image>",
    r"<\|image_pad\|>",
    r"<\|vision_start\|>",
    r"<\|vision_end\|>",
)

# Default pixel bounds for the Qwen processor (quirk ii).
# Per development_roadmap.md: min = 256*28*28 = 200704; working max = 1605632.
DEFAULT_MIN_PIXELS: int = 200_704
DEFAULT_MAX_PIXELS: int = 1_605_632

# HTTP policy (quirk iv).
REQUEST_TIMEOUT_S: float = 120.0
MAX_RETRIES: int = 1
RETRY_BACKOFF_S: float = 1.0
