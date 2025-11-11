"""
Abstract chunker interface.
"""
from typing import List, Protocol


class Chunker(Protocol):
    """Abstract chunker interface.

    Any concrete chunker must implement `split` and return a list of string
    chunks ready for embedding.
    """

    def split(self, text: str) -> List[str]:
        """Split raw text into chunks.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of chunk strings.
        """
