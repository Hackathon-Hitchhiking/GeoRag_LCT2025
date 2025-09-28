"""Клиент для взаимодействия с Ollama и моделью Qwen2.5-VL."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import httpx

LOG = logging.getLogger("georag.infrastructure.vlm.ollama")


@dataclass(frozen=True, slots=True)
class OllamaChatMessage:
    """Сообщение для Ollama chat API."""

    role: str
    content: str
    images: Sequence[str] | None = field(default=None)

    def to_payload(self) -> dict[str, Any]:
        """Преобразовать сообщение в словарь для HTTP API."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            payload["images"] = list(self.images)
        return payload


class OllamaVLMClient:
    """Асинхронный клиент Ollama для визуально-языковых моделей."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-vl:7b-instruct",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def chat(self, messages: Iterable[OllamaChatMessage]) -> str:
        """Отправить цепочку сообщений и вернуть текст ответа модели."""
        payload = {
            "model": self._model,
            "messages": [message.to_payload() for message in messages],
            "stream": False,
        }
        url = f"{self._base_url}/api/chat"
        LOG.debug(
            "event=ollama_chat_request model=%s url=%s message_count=%d",
            self._model,
            url,
            len(payload["messages"]),
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - сетевые ошибки логируем
            LOG.error("event=ollama_chat_failed error=%s", exc)
            raise
        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ответ Ollama не содержит текстового сообщения")
        LOG.debug("event=ollama_chat_response length=%d", len(content))
        return content

    async def describe_image(
        self,
        *,
        prompt: str,
        image_base64: str,
        system_prompt: str | None = None,
    ) -> str:
        """Сформировать ответ модели для одного изображения."""
        messages: list[OllamaChatMessage] = []
        if system_prompt:
            messages.append(OllamaChatMessage(role="system", content=system_prompt))
        messages.append(
            OllamaChatMessage(role="user", content=prompt, images=[image_base64])
        )
        return await self.chat(messages)
