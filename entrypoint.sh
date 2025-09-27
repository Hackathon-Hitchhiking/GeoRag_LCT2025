#!/usr/bin/env bash

set -e

exec uv run --no-dev uvicorn app.main:app --host 0.0.0.0 --port 8000
