"""FastAPI routes exposing ... functionality."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from ..logging import get_logger

LOG = get_logger("medical.api")

router = APIRouter(prefix="/v1", tags=["..."])