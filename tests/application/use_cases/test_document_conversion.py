"""
Unit tests for the DocumentConversionUseCase.
"""

import pathlib
from unittest.mock import MagicMock, call

import src.rag.application.use_cases.document_conversion as doc_use_case_mod
from src.rag.application.use_cases.document_conversion import (
    DocumentConversionUseCase,
)


def test_run_invokes_converter_and_logs(monkeypatch) -> None:
    """run() should log start/end and return converter outputs."""
    converter = MagicMock()
    outputs = [
        pathlib.Path("docs/a.md"),
        pathlib.Path("docs/b.md"),
    ]
    converter.convert_all.return_value = outputs
    logger = MagicMock()

    use_case = DocumentConversionUseCase(converter=converter, logger=logger)

    result = use_case.run()

    assert result == outputs
    converter.convert_all.assert_called_once_with()
    logger.info.assert_has_calls(
        [
            call("Starting document conversion use case"),
            call("Document conversion completed | written=%d", len(outputs)),
        ]
    )


def test_default_logger_is_resolved_from_factory(monkeypatch) -> None:
    """When no logger is provided, get_app_logger must be used."""
    stub_logger = MagicMock()
    monkeypatch.setattr(
        doc_use_case_mod, "get_app_logger", lambda: stub_logger
        )
    converter = MagicMock()

    use_case = DocumentConversionUseCase(converter=converter)

    assert use_case.logger is stub_logger
