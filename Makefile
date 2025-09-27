ifneq (,$(wildcard .env))
include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
endif

export PYTHONPATH := $(CURDIR)

.PHONY: lint typecheck ingest bootstrap-moscow

lint:
	uv run ruff check app scripts

typecheck:
	uv run mypy app scripts

ingest:
	uv run python scripts/ingest_train_data.py train_data

bootstrap-moscow:
	@test -n "$$MAPILLARY_TOKEN" || (echo "MAPILLARY_TOKEN is required" >&2; exit 1)
	uv run python scripts/download_moscow_mapillary.py --token "$$MAPILLARY_TOKEN" --output train_data/mapillary_moscow
