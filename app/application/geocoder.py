"""Интеграция с Nominatim для обратного геокодирования."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from geopy import Nominatim
from geopy.exc import GeocoderServiceError

LOG = logging.getLogger("georag.geocoder")


class Geocoder:
    """Обёртка над Nominatim с безопасной работой из async-кода."""

    def __init__(self, *, user_agent: str | None, timeout: int = 5) -> None:
        self._enabled = bool(user_agent)
        self._geocoder = (
            Nominatim(user_agent=user_agent, timeout=timeout) if user_agent else None
        )

    async def reverse(self, latitude: float | None, longitude: float | None) -> str | None:
        """Получить адрес по координатам, если сервис доступен."""
        if not self._enabled or latitude is None or longitude is None:
            return None

        def _reverse() -> str | None:
            try:
                assert self._geocoder is not None
                result = self._geocoder.reverse((latitude, longitude))
            except (AssertionError, GeocoderServiceError, ValueError) as exc:
                LOG.warning("event=geocode_failed error=%s", exc)
                return None
            if result is None:
                return None
            address: Any = result.address
            return str(address) if address else None

        return await asyncio.to_thread(_reverse)

    async def forward(self, query: str) -> tuple[float, float] | None:
        """Получить координаты по адресу, если сервис доступен."""
        if not self._enabled or not query.strip():
            return None

        def _forward() -> tuple[float, float] | None:
            try:
                assert self._geocoder is not None
                result = self._geocoder.geocode(query)
            except (AssertionError, GeocoderServiceError, ValueError) as exc:
                LOG.warning("event=geocode_forward_failed error=%s", exc)
                return None
            if result is None:
                return None
            latitude = getattr(result, "latitude", None)
            longitude = getattr(result, "longitude", None)
            if latitude is None or longitude is None:
                return None
            return float(latitude), float(longitude)

        return await asyncio.to_thread(_forward)
