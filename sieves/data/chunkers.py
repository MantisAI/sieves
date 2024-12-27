from typing import Any, Iterable, Protocol

import chonkie

from sieves.data.doc import Doc


class Chunker(Protocol):
    def __call__(
        self, docs: Iterable[Doc], include_meta: bool = False
    ) -> Iterable[Iterable[str]] | tuple[Iterable[Iterable[str]], Any]:
        """Chunks text in Doc objects.
        :param docs: Docs to chunk.
        :param include_meta: Whether to include meta information about the chunking process.
        :returns: Chunks per doc or chunks + meta info per doc.
        """


class Chonkie:
    """Wrapper for chonkie."""

    def __init__(self, chunker: chonkie.BaseChunker):
        self._chunker = chunker

    def __call__(
        self, docs: Iterable[Doc], include_meta: bool = False
    ) -> Iterable[Iterable[str]] | tuple[Iterable[Iterable[str]], Any]:
        """Uses Chonkie to chunk text in Doc objects.
        :param docs: Docs to chunk.
        :param include_meta: Whether to include meta information about the chunking process.
        :returns: Chunks per doc or chunks + meta info per doc.
        """
        chunks = self._chunker.chunk_batch([str(doc) for doc in docs])
        chunk_texts = [chunk.text for doc_chunks in chunks for chunk in doc_chunks]

        if include_meta:
            return chunk_texts, chunks
        return chunk_texts
