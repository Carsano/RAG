import logging
import sys
from pathlib import Path


class Logger:
    """Logger applicatif simple avec sortie console et fichier."""

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
        log_path = Path(log_file or (
            Path(__file__).resolve().parent / "app.log"
            ))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        if self.logger.handlers:
            return  # avoid duplicate handlers

        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)
        self.logger.addHandler(console)

        # File handler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)

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
