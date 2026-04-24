from __future__ import annotations

import logging
import os
import warnings


_IGNORED_WARNING_MESSAGES = (
    "`torch_dtype` is deprecated! Use `dtype` instead!",
    "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.",
)

_FILTERED_LOGGER_NAMES = (
    "transformers.configuration_utils",
    "transformers.modeling_utils",
    "transformers.dynamic_module_utils",
)


class _ExactMessageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in _IGNORED_WARNING_MESSAGES


def _configure_warning_filters() -> None:
    existing_filters = [entry for entry in os.environ.get("PYTHONWARNINGS", "").split(",") if entry]
    for message in _IGNORED_WARNING_MESSAGES:
        warnings.filterwarnings("ignore", message=message)
        filter_spec = f"ignore:{message}"
        if filter_spec not in existing_filters:
            existing_filters.append(filter_spec)
    if existing_filters:
        os.environ["PYTHONWARNINGS"] = ",".join(existing_filters)


def _configure_logging_filters() -> None:
    for logger_name in _FILTERED_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        if getattr(logger, "_em_exact_message_filter_installed", False):
            continue
        logger.addFilter(_ExactMessageFilter())
        logger._em_exact_message_filter_installed = True


def configure_eval_environment() -> None:
    transformers_cache = os.environ.pop("TRANSFORMERS_CACHE", None)
    if transformers_cache:
        os.environ.setdefault("HF_HOME", transformers_cache)
    _configure_warning_filters()
    _configure_logging_filters()


def configure_vllm_environment() -> None:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _configure_warning_filters()
    _configure_logging_filters()
