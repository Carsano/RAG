"""Logging module providing configurable for the RAG application.

This module defines a LoggerBuilder with a fluent interface
and concrete loggers for application and usage tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from src.rag.utils.utils import get_project_root


@dataclass
class LoggerConfig:
    """Configuration data container for logger settings.

    Attributes:
        name (str): Name of the logger.
        subdir (str): Subdirectory under the logs root directory for log files.
        file_prefix (str): Prefix for the log file names.
        console (bool): Whether to enable console logging.
        level (int): Logging level (e.g., logging.INFO).
    """
    name: str = "app"
    subdir: str = ""
    file_prefix: str = "RAG_logs"
    console: bool = True
    level: int = logging.INFO


class LoggerBuilder:
    """Fluent builder for configured loggers.

    Provides methods for configuring logger properties and building a
    configured logging.Logger instance. Supports dependency injection
    for handlers and formatters to facilitate testing and customization.

    Attributes:
        _cfg (LoggerConfig): Configuration for the logger.
        _formatter_factory (callable): Factory function to create a formatter.
        _file_handler_factory (callable): Factory function to create
                                          a file handler.
        _console_handler_factory (callable): Factory function to create
                                             a console handler.
    """

    def __init__(self, config: LoggerConfig | None = None):
        """Initialize the LoggerBuilder with an optional configuration.

        Args:
            config (LoggerConfig | None): Optional initial configuration.
        """
        self._cfg = config or LoggerConfig()
        self._formatter_factory = self._default_formatter
        self._file_handler_factory = self._default_file_handler
        self._console_handler_factory = self._default_console_handler

    def name(self, logger_name: str) -> "LoggerBuilder":
        """Set the logger name.

        Args:
            logger_name (str): Name to assign to the logger.

        Returns:
            LoggerBuilder: The builder instance (for chaining).
        """
        self._cfg.name = logger_name
        return self

    def subdir(self, log_subdir: str) -> "LoggerBuilder":
        """Set the subdirectory for log files.

        Args:
            log_subdir (str): Subdirectory name under the logs root.

        Returns:
            LoggerBuilder: The builder instance (for chaining).
        """
        self._cfg.subdir = log_subdir
        return self

    def prefix(self, file_prefix: str) -> "LoggerBuilder":
        """Set the prefix for log file names.

        Args:
            file_prefix (str): Prefix string for log files.

        Returns:
            LoggerBuilder: The builder instance (for chaining).
        """
        self._cfg.file_prefix = file_prefix
        return self

    def console(self, enabled: bool) -> "LoggerBuilder":
        """Enable or disable console logging.

        Args:
            enabled (bool): True to enable console logging, False to disable.

        Returns:
            LoggerBuilder: The builder instance (for chaining).
        """
        self._cfg.console = enabled
        return self

    def level(self, level: int) -> "LoggerBuilder":
        """Set the logging level.

        Args:
            level (int): Logging level (e.g., logging.INFO).

        Returns:
            LoggerBuilder: The builder instance (for chaining).
        """
        self._cfg.level = level
        return self

    def formatter(self, factory) -> "LoggerBuilder":
        """Set a custom formatter factory.

        Args:
            factory (callable): Factory function that returns
            a logging.Formatter.

        Returns:
            LoggerBuilder: The builder instance (for chaining).
        """
        self._formatter_factory = factory
        return self

    def file_handler(self, factory) -> "LoggerBuilder":
        """Set a custom file handler factory.

        Args:
            factory (callable): Factory function that takes (Path, Formatter)
            and returns a logging.Handler.

        Returns:
            LoggerBuilder: The builder instance (for chaining).
        """
        self._file_handler_factory = factory
        return self

    def console_handler(self, factory) -> "LoggerBuilder":
        """Set a custom console handler factory.

        Args:
            factory (callable): Factory function that takes a Formatter
            and returns a logging.Handler.

        Returns:
            LoggerBuilder: The builder instance (for chaining).
        """
        self._console_handler_factory = factory
        return self

    def build(self) -> logging.Logger:
        """Build and return the configured logger instance.

        Creates the log directory if needed, sets up handlers and formatter,
        and returns a logger with the configured name and level.
        Ensures idempotency.

        Returns:
            logging.Logger: The configured logger instance.
        """
        log_dir = self._ensure_dir((self._log_root() / self._cfg.subdir)
                                   if self._cfg.subdir
                                   else self._ensure_dir(self._log_root()))
        log_path = (log_dir /
                    f"{self._today_stamp()}_{self._cfg.file_prefix}.log")

        logger = logging.getLogger(self._cfg.name)
        if logger.handlers:
            return logger  # idempotent

        fmt = self._formatter_factory()
        if self._cfg.console:
            logger.addHandler(self._console_handler_factory(fmt))
        logger.addHandler(self._file_handler_factory(log_path, fmt))
        logger.setLevel(self._cfg.level)
        return logger

    @staticmethod
    def _log_root() -> Path:
        """Return the root directory path for logs.

        Returns:
            Path: Path object pointing to the logs root directory.
        """
        return get_project_root() / "logs"

    @staticmethod
    def _ensure_dir(p: Path) -> Path:
        """Ensure that the given directory path exists.

        Creates the directory and any necessary parents if they do not exist.

        Args:
            p (Path): Directory path to ensure.

        Returns:
            Path: The same Path object passed in.
        """
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _today_stamp() -> str:
        """Return today's date formatted as YYYYMMDD.

        Returns:
            str: Date string in YYYYMMDD format.
        """
        return datetime.now().strftime("%Y%m%d")

    @staticmethod
    def _default_formatter() -> logging.Formatter:
        """Create the default logging formatter.

        Returns:
            logging.Formatter: Formatter with timestamp, level, and message.
        """
        return logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    @staticmethod
    def _default_file_handler(path: Path,
                              fmt: logging.Formatter) -> logging.Handler:
        """Create the default file handler.

        Args:
            path (Path): Path to the log file.
            fmt (logging.Formatter): Formatter to apply to the handler.

        Returns:
            logging.Handler: Configured FileHandler instance.
        """
        h = logging.FileHandler(path, encoding="utf-8")
        h.setLevel(logging.INFO)
        h.setFormatter(fmt)
        return h

    @staticmethod
    def _default_console_handler(fmt: logging.Formatter) -> logging.Handler:
        """Create the default console handler.

        Args:
            fmt (logging.Formatter): Formatter to apply to the handler.

        Returns:
            logging.Handler: Configured StreamHandler instance for stdout.
        """
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(fmt)
        return h


class Logger:
    """Simple application logger with console and file output.

    Implements a singleton pattern to provide a single logger instance
    throughout the application.

    Attributes:
        _instance (Logger | None): Singleton instance of Logger.
        logger (logging.Logger): The underlying logging.Logger instance.
    """

    _instance = None

    def __new__(cls, name: str = "app", log_file: str | None = None):
        """Create or return the singleton instance of Logger.

        Args:
            name (str): The name of the logger.
            log_file (str | None): Path to the log file.

        Returns:
            Logger: The singleton Logger instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(name, log_file)
        return cls._instance

    def _initialize(self, name: str, log_file: str | None = None):
        """Initialize the logger with console and file handlers.

        Args:
            name (str): The name of the logger.
            log_file (str | None): Path to the log file.
        """
        self.logger = (
            LoggerBuilder()
            .name(name)
            .subdir("")
            .prefix("RAG_logs")
            .console(True)
            .build()
        )

    def info(self, msg: str):
        """Log a message with INFO level.

        Args:
            msg (str): The message to log.
        """
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log a message with WARNING level.

        Args:
            msg (str): The message to log.
        """
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log a message with ERROR level.

        Args:
            msg (str): The message to log.
        """
        self.logger.error(msg)

    def debug(self, msg: str):
        """Log a message with DEBUG level.

        Args:
            msg (str): The message to log.
        """
        self.logger.debug(msg)

    def critical(self, msg: str):
        """Log a message with CRITICAL level.

        Args:
            msg (str): The message to log.
        """
        self.logger.critical(msg)


class AppLogger(Logger):
    """Logger dedicated to the application with a distinct log file.

    Implements a singleton pattern to provide a single application
    logger instance.

    Attributes:
        _instance (AppLogger | None): Singleton instance of AppLogger.
    """
    _instance = None

    def _initialize(self, name: str = "app", log_file: str | None = None):
        """Initialize the application logger with appropriate handlers.

        Args:
            name (str): Logger name.
            log_file (str | None): Optional explicit log file path.
        """
        self.logger = (
            LoggerBuilder()
            .name("app")
            .subdir("app")
            .prefix("App_logs")
            .console(True)
            .build()
        )


class UsageLogger(Logger):
    """Logger dedicated to usage tracking, isolated from the main logger.

    Implements a singleton pattern to provide a single usage logger instance.

    Attributes:
        _instance (UsageLogger | None): Singleton instance of UsageLogger.
    """
    _instance = None

    def _initialize(self, name: str = "usage", log_file: str | None = None):
        """Initialize the usage logger with appropriate handlers.

        Args:
            name (str): Logger name.
            log_file (str | None): Optional explicit log file path.
        """
        self.logger = (
            LoggerBuilder()
            .name("usage")
            .subdir("usages")
            .prefix("Usages_logs")
            .console(False)
            .build()
        )


def get_app_logger():
    """Return the singleton instance of AppLogger.

    Returns:
        AppLogger: Logger configured for general application logging.
    """
    return AppLogger("app")


def get_usage_logger():
    """Return the singleton instance of UsageLogger.

    Returns:
        UsageLogger: Logger configured for tracking usage events.
    """
    return UsageLogger("usage")
