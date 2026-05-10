"""Image-aware history compressor: ports open-webui's qwenvl_image_compress
filter into our backend's service layer.
"""
from app.services.image_compressor.engine import ImageCompressorEngine

__all__ = ["ImageCompressorEngine"]
