"""
CLI entry point for converting source documentation to Markdown.
"""

from src.rag.application.use_cases.document_conversion import (
    DocumentConversionUseCase,
)
from src.rag.infrastructure.converters.converters import DocumentConverter
from src.rag.utils.utils import get_project_root


def main() -> None:
    """Run the document conversion pipeline."""
    project_root = get_project_root()
    input_root = project_root / "data" / "controlled_documentation"
    output_root = project_root / "data" / "clean_md_database"

    converter = DocumentConverter(
        input_root=input_root,
        output_root=output_root,
    )
    use_case = DocumentConversionUseCase(converter=converter)
    use_case.run()


if __name__ == "__main__":
    main()
