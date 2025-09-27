"""Модуль логирования."""

import logging
import sys
import time


class _LogfmtFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        msg = record.getMessage().replace("\n", " ")
        return f'ts={ts} level={record.levelname} logger={record.name} msg="{msg}"'


def configure_logging(service_name: str, level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_LogfmtFormatter())
    root.addHandler(h)
    logging.getLogger(service_name).setLevel(
        getattr(logging, level.upper(), logging.INFO)
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
