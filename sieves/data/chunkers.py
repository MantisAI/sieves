from typing import Any, Iterable

import chonkie

from sieves.data import Doc


class Chonkie:
    """Wrapper for chonkie."""

    def __init__(self, chunker: chonkie.BaseChunker):
        self._chunker = chunker

    def __call__(
        self, docs: Iterable[Doc], include_meta: bool = False
    ) -> Iterable[Iterable[str]] | tuple[Iterable[Iterable[str]], Any]:
        chunks = self._chunker.chunk_batch([str(doc) for doc in docs])
        chunk_texts = [chunk.text for doc_chunks in chunks for chunk in doc_chunks]

        if include_meta:
            return chunk_texts, chunks
        return chunk_texts
