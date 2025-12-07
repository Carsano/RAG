"""
Use case orchestrating documentation conversion to Markdown.
"""

import logging
import pathlib
from typing import List, Optional

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

    def run(self) -> List[pathlib.Path]:
        """Convert every supported document and return generated paths."""
        self.logger.info("Starting document conversion use case")
        outputs = self.converter.convert_all()
        self.logger.info(
            f"Document conversion completed | written={len(outputs)}"
        )
        return outputs

    def run_for_files(self, paths: List[pathlib.Path]) -> List[pathlib.Path]:
        """Convert only the provided files."""
        self.logger.info("Starting document conversion use case")
        self.logger.info(
            f"Converting selected documents | count={len(paths)}"
        )
        outputs = self.converter.convert_paths(paths)
        self.logger.info(
            f"Document conversion completed | written={len(outputs)}"
        )
        return outputs


__all__ = [
    "DocumentConversionUseCase"
    ]
