"""IndicTrans2-token-counted chunking with context kept outside source text."""

import logging
import math
import uuid
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import List, Optional, Protocol, Sequence

from .models import PreprocessedDocument, Sentence, TranslationChunk

logger = logging.getLogger(__name__)

INDICTRANS2_TOKENIZER_CHECKPOINT = "ai4bharat/indictrans2-en-indic-1B"


class TextTokenizer(Protocol):
    """The minimal tokenizer contract needed for exact counting and context."""

    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...


class IndicTrans2Tokenizer:
    """Thin wrapper around the English-to-Indic IndicTrans2 SentencePiece tokenizer."""

    def __init__(self, tokenizer=None, target_language: str = "hin_Deva") -> None:
        self._tokenizer = tokenizer or self._load_tokenizer()
        self.target_language = target_language

    def encode(self, text: str) -> List[int]:
        """Count English source subwords and IndicTrans2's language-control tokens."""
        model_input = "eng_Latn {} {}".format(self.target_language, text)
        return list(self._tokenizer.encode(model_input, add_special_tokens=False, verbose=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode source subwords using the English vocabulary, not target-side IDs."""
        source_pieces = [
            self._tokenizer.src_decoder.get(token_id, self._tokenizer.unk_token)
            for token_id in token_ids
        ]
        return self._tokenizer.src_spm.DecodePieces(source_pieces).strip()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_tokenizer():
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Chunking requires transformers and sentencepiece; install requirements.txt"
            ) from error
        try:
            return AutoTokenizer.from_pretrained(
                INDICTRANS2_TOKENIZER_CHECKPOINT,
                trust_remote_code=True,
            )
        except OSError as error:
            if "gated" in str(error).casefold() or "403" in str(error):
                raise RuntimeError(
                    "English-source chunking requires access to the gated checkpoint "
                    "ai4bharat/indictrans2-en-indic-1B. Request access on Hugging Face, "
                    "then authenticate with hf auth login."
                ) from error
            raise


class TokenBudgetManager:
    """Reserve capacity for overlap so context never pushes a chunk over its cap."""

    def __init__(self, max_tokens: int, reserve_fraction: float = 0.12) -> None:
        if max_tokens < 3:
            raise ValueError("max_tokens must be at least 3 to reserve context safely")
        if not 0.0 <= reserve_fraction < 1.0:
            raise ValueError("reserve_fraction must be between 0 and 1")
        self.max_tokens = max_tokens
        self.reserved_tokens = min(max_tokens - 1, int(math.ceil(max_tokens * reserve_fraction)))
        self.content_budget = max_tokens - self.reserved_tokens
        self.overlap_tokens = self.reserved_tokens // 2


@dataclass
class ChunkDraft:
    """Unpersisted chunk content before overlap and database IDs are assigned."""

    section_id: str
    paragraph_id: str
    src_text: str


@dataclass
class ChunkWithContext:
    """Chunk draft with context fields kept separate from translatable text."""

    draft: ChunkDraft
    preceding_context: Optional[str]
    following_context: Optional[str]


class SemanticChunker:
    """Group complete English sentences, favoring whole paragraph boundaries."""

    def chunk(
        self,
        document: PreprocessedDocument,
        tokenizer: TextTokenizer,
        content_budget: int,
    ) -> List[ChunkDraft]:
        """Pack paragraphs when possible and split oversized paragraphs only by sentence."""
        drafts: List[ChunkDraft] = []
        current_sentences: List[Sentence] = []
        current_section: Optional[str] = None

        def text_of(sentences: Sequence[Sentence]) -> str:
            return " ".join(sentence.text for sentence in sentences)

        def flush() -> None:
            nonlocal current_sentences, current_section
            if current_sentences:
                drafts.append(
                    ChunkDraft(
                        section_id=current_sentences[0].section_id,
                        paragraph_id=current_sentences[0].paragraph_id,
                        src_text=text_of(current_sentences),
                    )
                )
            current_sentences = []
            current_section = None

        for paragraph in document.paragraphs:
            paragraph_sentences = paragraph.sentences
            if not paragraph_sentences:
                continue
            if current_section is not None and paragraph.section_id != current_section:
                flush()

            paragraph_text = text_of(paragraph_sentences)
            paragraph_count = len(tokenizer.encode(paragraph_text))

            if paragraph_count <= content_budget:
                combined = current_sentences + paragraph_sentences
                if current_sentences and len(tokenizer.encode(text_of(combined))) > content_budget:
                    flush()
                    combined = list(paragraph_sentences)
                current_sentences = combined
                current_section = paragraph.section_id
                continue

            flush()
            paragraph_group: List[Sentence] = []
            for sentence in paragraph_sentences:
                sentence_count = len(tokenizer.encode(sentence.text))
                if sentence_count > content_budget:
                    raise ValueError(
                        "Sentence {} has {} tokens, exceeding the {}-token content budget; "
                        "sentences are never split".format(
                            sentence.sentence_id, sentence_count, content_budget
                        )
                    )
                candidate = paragraph_group + [sentence]
                if paragraph_group and len(tokenizer.encode(text_of(candidate))) > content_budget:
                    drafts.append(
                        ChunkDraft(
                            section_id=paragraph_group[0].section_id,
                            paragraph_id=paragraph_group[0].paragraph_id,
                            src_text=text_of(paragraph_group),
                        )
                    )
                    paragraph_group = [sentence]
                else:
                    paragraph_group = candidate
            if paragraph_group:
                drafts.append(
                    ChunkDraft(
                        section_id=paragraph_group[0].section_id,
                        paragraph_id=paragraph_group[0].paragraph_id,
                        src_text=text_of(paragraph_group),
                    )
                )

        flush()
        return drafts


class OverlapInjector:
    """Build neighboring context from model tokens without altering srcText."""

    def inject(
        self,
        drafts: Sequence[ChunkDraft],
        tokenizer: TextTokenizer,
        overlap_tokens: int,
        max_tokens: int,
    ) -> List[ChunkWithContext]:
        """Attach adjacent token windows, trimming decoded text to the true max."""
        result = []
        for index, draft in enumerate(drafts):
            previous_ids = (
                tokenizer.encode(drafts[index - 1].src_text)[2:][-overlap_tokens:]
                if index > 0 and overlap_tokens
                else []
            )
            following_ids = (
                tokenizer.encode(drafts[index + 1].src_text)[2 : 2 + overlap_tokens]
                if index + 1 < len(drafts) and overlap_tokens
                else []
            )

            preceding_context = tokenizer.decode(previous_ids) if previous_ids else None
            following_context = tokenizer.decode(following_ids) if following_ids else None
            while self._combined_count(tokenizer, preceding_context, draft.src_text, following_context) > max_tokens:
                if previous_ids and (len(previous_ids) >= len(following_ids) or not following_ids):
                    previous_ids = previous_ids[1:]
                    preceding_context = tokenizer.decode(previous_ids) if previous_ids else None
                elif following_ids:
                    following_ids = following_ids[:-1]
                    following_context = tokenizer.decode(following_ids) if following_ids else None
                else:
                    raise ValueError("Chunk source text exceeds max_tokens after context removal")

            result.append(ChunkWithContext(draft, preceding_context, following_context))
        return result

    @staticmethod
    def _combined_count(
        tokenizer: TextTokenizer,
        preceding_context: Optional[str],
        src_text: str,
        following_context: Optional[str],
    ) -> int:
        combined = " ".join(part for part in (preceding_context, src_text, following_context) if part)
        return len(tokenizer.encode(combined))


class ChunkBuilder:
    """Build persisted-shape chunk rows in stable, zero-based document order."""

    def build(self, document_id: str, chunks: Sequence[ChunkWithContext]) -> List[TranslationChunk]:
        """Assign IDs and workflow defaults only after chunk boundaries are final."""
        return [
            TranslationChunk(
                chunkId=str(uuid.uuid4()),
                documentId=document_id,
                sectionId=chunk.draft.section_id,
                paragraphId=chunk.draft.paragraph_id,
                unitIndex=index,
                srcText=chunk.draft.src_text,
                precedingContext=chunk.preceding_context,
                followingContext=chunk.following_context,
                status="pending",
                repairCount=0,
            )
            for index, chunk in enumerate(chunks)
        ]


def chunk_document(
    preprocessed: PreprocessedDocument,
    max_tokens: int,
    tokenizer: Optional[TextTokenizer] = None,
) -> List[TranslationChunk]:
    """Tokenize, budget, semantically group, overlap, and build translation chunks.

    Production calls use the gated IndicTrans2 English-to-Indic tokenizer checkpoint.
    The injectable tokenizer exists for deterministic unit tests and local probes.
    """
    started = perf_counter()
    active_tokenizer = tokenizer or IndicTrans2Tokenizer(
        target_language=preprocessed.document.targetLanguage
    )
    budget = TokenBudgetManager(max_tokens)
    drafts = SemanticChunker().chunk(preprocessed, active_tokenizer, budget.content_budget)
    contextualized = OverlapInjector().inject(
        drafts,
        active_tokenizer,
        budget.overlap_tokens,
        budget.max_tokens,
    )
    chunks = ChunkBuilder().build(preprocessed.document.documentId, contextualized)
    logger.info(
        "Chunking completed in %.3fs: %d chunks (content budget %d/%d tokens)",
        perf_counter() - started,
        len(chunks),
        budget.content_budget,
        budget.max_tokens,
    )
    return chunks