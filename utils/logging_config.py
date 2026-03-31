"""
Structured Logging Framework for BddAgent Experiments

Provides JSON-structured logging with experiment context,
separate streams for different log levels, and easy analysis.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def __init__(self, experiment_id: Optional[str] = None):
        super().__init__()
        self.experiment_id = experiment_id

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.experiment_id:
            log_entry["experiment_id"] = self.experiment_id

        # Add extra fields if present
        if hasattr(record, 'namespace'):
            log_entry["namespace"] = record.namespace
        if hasattr(record, 'test_index'):
            log_entry["test_index"] = record.test_index
        if hasattr(record, 'model'):
            log_entry["model"] = record.model
        if hasattr(record, 'error_type'):
            log_entry["error_type"] = record.error_type

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra data
        if hasattr(record, 'data') and record.data:
            log_entry["data"] = record.data

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for human-readable output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        # Build prefix
        prefix = f"{color}[{timestamp}] {record.levelname:8}{self.RESET}"

        # Add context if available
        context = ""
        if hasattr(record, 'namespace'):
            context = f" [{record.namespace}]"
        elif hasattr(record, 'test_index'):
            context = f" [test {record.test_index}]"

        message = record.getMessage()

        return f"{prefix}{context} {message}"


@dataclass
class LogContext:
    """Context for structured logging."""
    experiment_id: Optional[str] = None
    namespace: Optional[str] = None
    test_index: Optional[int] = None
    model: Optional[str] = None


class ExperimentLogger:
    """
    Experiment-aware logger that adds context to all log messages.
    """

    def __init__(
        self,
        name: str,
        experiment_id: Optional[str] = None,
        log_dir: Optional[str] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        self.name = name
        self.experiment_id = experiment_id
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.context = LogContext(experiment_id=experiment_id)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(ConsoleFormatter())
        self.logger.addHandler(console_handler)

        # File handlers (if log_dir specified)
        if log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_prefix = f"{experiment_id}_" if experiment_id else ""

            # JSON log file (all levels)
            json_handler = logging.FileHandler(
                self.log_dir / f"{exp_prefix}experiment_{timestamp}.jsonl",
                encoding='utf-8'
            )
            json_handler.setLevel(file_level)
            json_handler.setFormatter(JSONFormatter(experiment_id))
            self.logger.addHandler(json_handler)

            # Error-only file
            error_handler = logging.FileHandler(
                self.log_dir / f"{exp_prefix}errors_{timestamp}.jsonl",
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(JSONFormatter(experiment_id))
            self.logger.addHandler(error_handler)

    def set_context(
        self,
        namespace: Optional[str] = None,
        test_index: Optional[int] = None,
        model: Optional[str] = None
    ):
        """Set logging context for subsequent messages."""
        if namespace is not None:
            self.context.namespace = namespace
        if test_index is not None:
            self.context.test_index = test_index
        if model is not None:
            self.context.model = model

    def clear_context(self):
        """Clear logging context."""
        self.context = LogContext(experiment_id=self.experiment_id)

    def _add_context(self, extra: Optional[Dict] = None) -> Dict:
        """Add current context to extra dict."""
        result = {}
        if self.context.namespace:
            result['namespace'] = self.context.namespace
        if self.context.test_index is not None:
            result['test_index'] = self.context.test_index
        if self.context.model:
            result['model'] = self.context.model
        if extra:
            result.update(extra)
        return result

    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, extra=self._add_context(kwargs))

    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, extra=self._add_context(kwargs))

    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, extra=self._add_context(kwargs))

    def error(self, msg: str, error_type: Optional[str] = None, **kwargs):
        """Log error message."""
        extra = self._add_context(kwargs)
        if error_type:
            extra['error_type'] = error_type
        self.logger.error(msg, extra=extra)

    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, extra=self._add_context(kwargs))

    def exception(self, msg: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(msg, extra=self._add_context(kwargs))

    def test_start(self, namespace: str, test_index: int):
        """Log test case start."""
        self.set_context(namespace=namespace, test_index=test_index)
        self.info(f"Starting test case: {namespace}")

    def test_end(self, success: bool, duration: float):
        """Log test case end."""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"Test completed: {status} ({duration:.2f}s)")
        self.clear_context()

    def experiment_start(self, config: Dict[str, Any]):
        """Log experiment start."""
        self.info(f"Starting experiment: {self.experiment_id}", data=config)

    def experiment_end(self, summary: Dict[str, Any]):
        """Log experiment end."""
        self.info(f"Experiment completed: {self.experiment_id}", data=summary)


# Global logger instance
_default_logger: Optional[ExperimentLogger] = None


def setup_logging(
    experiment_id: Optional[str] = None,
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> ExperimentLogger:
    """
    Set up experiment logging.

    Args:
        experiment_id: Unique experiment identifier
        log_dir: Directory for log files
        console_level: Console logging level
        file_level: File logging level

    Returns:
        ExperimentLogger instance
    """
    global _default_logger
    _default_logger = ExperimentLogger(
        name="bddagent",
        experiment_id=experiment_id,
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level
    )
    return _default_logger


def get_logger() -> ExperimentLogger:
    """Get the default logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger
