"""
Use case orchestrating documentation conversion to Markdown.
"""

import logging
from typing import Optional

from src.rag.application.ports.converters import DocumentConversionService
from src.rag.infrastructure.logging.logger import get_app_logger


class DocumentConversionUseCase:
    """
    Application service that triggers document conversion.

    Keeps infrastructure-specific logic inside the converter adapter and only
    coordinates the high-level workflow from the application layer.
    """

    def __init__(
        self,
        converter: DocumentConversionService,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the use case.

        Args:
            converter (DocumentConversionService): Adapter handling the
                conversion work.
            logger (Optional[logging.Logger]): Application logger instance.
        """
        self.converter = converter
        self.logger = logger or get_app_logger()
