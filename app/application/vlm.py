"""Сервис для получения описания сцены с помощью визуально-языковой модели."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.infrastructure.vlm import OllamaVLMClient

LOG = logging.getLogger("georag.application.vlm")


@dataclass(slots=True)
class SceneDescription:
    """Структурированное описание местности, возвращаемое моделью."""

    description: str
    address: str | None = None


class VisionLanguageAnalyzer:
    """Высокоуровневый интерфейс описания сцен с помощью VLM."""

    _SYSTEM_PROMPT = (
        "You are a helpful assistant that analyses street-level imagery and "
        "returns structured JSON."
    )
    _USER_PROMPT = (
        "You receive a single outdoor street-level photo. Carefully inspect the image and "
        "prepare a concise JSON response with two fields: 'address' and 'description'. "
        "The 'address' must be null when there is no explicit postal address, signage or "
        "text pointing to a precise location. When such hints exist, extract a short "
        "address string in the original language. The 'description' should briefly "
        "summarize the surroundings, notable objects, road type, weather and other "
        "geographical cues. Answer strictly in valid JSON without additional text."
    )

    def __init__(self, client: OllamaVLMClient) -> None:
        self._client = client

    async def describe(self, image_base64: str) -> SceneDescription:
        """Получить описание сцены для изображения в base64."""
        raw_response = await self._client.describe_image(
            prompt=self._USER_PROMPT,
            image_base64=image_base64,
            system_prompt=self._SYSTEM_PROMPT,
        )
        payload = self._extract_json(raw_response)
        description = payload.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Модель вернула некорректное описание")
        address_value = payload.get("address")
        if isinstance(address_value, str):
            address = address_value.strip() or None
        else:
            address = None
        scene = SceneDescription(description=description.strip(), address=address)
        LOG.debug("event=vlm_scene_description address_present=%s", bool(address))
        return scene

    @classmethod
    def from_settings(cls, settings: Settings) -> "VisionLanguageAnalyzer":
        """Создать сервис, используя конфигурацию приложения."""
        client = OllamaVLMClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout,
        )
        return cls(client)

    @staticmethod
    def _extract_json(raw_response: str) -> dict[str, object]:
        """Достать JSON из произвольного текстового ответа модели."""
        decoder = json.JSONDecoder()
        index = raw_response.find("{")
        while index != -1:
            try:
                payload, offset = decoder.raw_decode(raw_response[index:])
            except json.JSONDecodeError:
                index = raw_response.find("{", index + 1)
                continue
            if isinstance(payload, dict):
                return payload
            index = raw_response.find("{", index + offset)
        raise ValueError("Не удалось распарсить JSON из ответа модели")
