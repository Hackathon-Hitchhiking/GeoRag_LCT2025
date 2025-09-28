"""Инфраструктурные клиенты для визуально-языковых моделей."""

from .ollama import OllamaChatMessage, OllamaVLMClient

__all__ = [
    "OllamaChatMessage",
    "OllamaVLMClient",
]
