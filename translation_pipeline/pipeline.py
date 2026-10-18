"""Async orchestration for one-time preprocessing and sequential chunk handoff."""

import asyncio
import logging
from typing import List, Optional

from .chunking import TextTokenizer, chunk_document
from .models import Entity, TranslationChunk
from .preprocessing import (
    EntityCandidateExtractor,
    LanguageExtractor,
    preprocess_document,
)

logger = logging.getLogger(__name__)


class TranslationChunkQueue(asyncio.Queue):
    """An asyncio queue carrying ordered chunks and their document entity registry."""

    def __init__(self, chunks: List[TranslationChunk], entities: List[Entity]) -> None:
        super().__init__()
        self.entities = list(entities)
        for chunk in sorted(chunks, key=lambda item: item.unitIndex):
            self.put_nowait(chunk)


async def run_preprocessing_pipeline(
    file_path: str,
    document_id: str,
    max_tokens: int,
    target_language: str = "hin_Deva",
    genre: Optional[str] = None,
    register: Optional[str] = None,
    tokenizer: Optional[TextTokenizer] = None,
    language_detector: Optional[LanguageExtractor] = None,
    entity_extractor: Optional[EntityCandidateExtractor] = None,
    terminology_extractor: Optional[EntityCandidateExtractor] = None,
) -> TranslationChunkQueue:
    """Run English preprocessing once and return its ordered translation chunks.

    ``queue.entities`` is the merged Entity + Term Registry for the same document.
    Consumers should use ``await queue.get()`` and call ``queue.task_done()`` after
    each chunk, in the order established by ``unitIndex``.
    """
    preprocessed = await asyncio.to_thread(
        preprocess_document,
        file_path,
        document_id,
        target_language,
        genre,
        register,
        language_detector,
        entity_extractor,
        terminology_extractor,
    )
    chunks = await asyncio.to_thread(chunk_document, preprocessed, max_tokens, tokenizer)
    queue = TranslationChunkQueue(chunks, preprocessed.entities)
    logger.info(
        "Pipeline handoff ready: %d chunks and %d entities for document %s",
        queue.qsize(),
        len(queue.entities),
        document_id,
    )
    return queue