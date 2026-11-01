"""IndicTrans2-token-counted chunking with context kept outside source text."""

import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import List, Optional, Protocol, Sequence, Tuple

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
        self.total_tokens = 0
        self.unknown_tokens = 0
        try:
            from sacremoses import MosesPunctNormalizer, MosesTokenizer

            self._mpn = MosesPunctNormalizer(lang="en")
            self._mt = MosesTokenizer(lang="en")
        except ImportError:
            self._mpn = None
            self._mt = None

    def _preprocess(self, text: str) -> str:
        if self._mpn and self._mt and text:
            normalized = self._mpn.normalize(text)
            return self._mt.tokenize(normalized, return_str=True)
        return text

    def encode(self, text: str) -> List[int]:
        """Count English source subwords and IndicTrans2's language-control tokens."""
        preprocessed = self._preprocess(text)
        model_input = "eng_Latn {} {}".format(self.target_language, preprocessed)
        token_ids = list(self._tokenizer.encode(model_input, add_special_tokens=False, verbose=False))
        unknown_id = getattr(self._tokenizer, "unk_token_id", None)
        self.total_tokens += len(token_ids)
        if unknown_id is not None:
            self.unknown_tokens += sum(token_id == unknown_id for token_id in token_ids)
        return token_ids

    @property
    def unknown_token_rate(self) -> float:
        """Report unknown-token share over all actual model-tokenizer calls."""
        return self.unknown_tokens / self.total_tokens if self.total_tokens else 0.0

    def text_token_count(self, text: str) -> int:
        """Count content pieces without the two IndicTrans2 language tags."""
        return max(0, len(self.encode(text)) - 2)

    def measure(self, text: str) -> Tuple[int, float]:
        """Return content subword count and unknown-token rate for one text span."""
        token_ids = self.encode(text)
        unknown_id = getattr(self._tokenizer, "unk_token_id", None)
        unknown_count = sum(token_id == unknown_id for token_id in token_ids) if unknown_id is not None else 0
        return max(0, len(token_ids) - 2), unknown_count / len(token_ids) if token_ids else 0.0

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
    sentences: List[Sentence] = field(default_factory=list)
    source_start: Optional[int] = None
    source_end: Optional[int] = None


@dataclass
class ChunkWithContext:
    """Chunk draft with context fields kept separate from translatable text."""

    draft: ChunkDraft
    preceding_context: Optional[str]
    following_context: Optional[str]
    content_token_count: int = 0
    context_token_count: int = 0


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
            start = sentences[0].char_start
            end = sentences[-1].char_end
            if 0 <= start < end <= len(document.normalized_text):
                return document.normalized_text[start:end]
            return " ".join(sentence.text.strip() for sentence in sentences)

        def flush() -> None:
            nonlocal current_sentences, current_section
            if current_sentences:
                drafts.append(
                    ChunkDraft(
                        section_id=current_sentences[0].section_id,
                        paragraph_id=current_sentences[0].paragraph_id,
                        src_text=text_of(current_sentences),
                        sentences=list(current_sentences),
                        source_start=current_sentences[0].char_start,
                        source_end=current_sentences[-1].char_end,
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
                    split_sentences = self._split_oversized_sentence(sentence, tokenizer, content_budget)
                    if not split_sentences:
                        raise ValueError(
                            "Sentence {} has {} tokens and no safe semicolon/colon split fits "
                            "the {}-token content budget".format(
                                sentence.sentence_id, sentence_count, content_budget
                            )
                        )
                else:
                    split_sentences = [sentence]

                for split_sentence in split_sentences:
                    candidate = paragraph_group + [split_sentence]
                    if paragraph_group and len(tokenizer.encode(text_of(candidate))) > content_budget:
                        drafts.append(
                            ChunkDraft(
                                section_id=paragraph_group[0].section_id,
                                paragraph_id=paragraph_group[0].paragraph_id,
                                src_text=text_of(paragraph_group),
                                sentences=list(paragraph_group),
                                source_start=paragraph_group[0].char_start,
                                source_end=paragraph_group[-1].char_end,
                            )
                        )
                        paragraph_group = [split_sentence]
                    else:
                        paragraph_group = candidate
            if paragraph_group:
                drafts.append(
                    ChunkDraft(
                        section_id=paragraph_group[0].section_id,
                        paragraph_id=paragraph_group[0].paragraph_id,
                        src_text=text_of(paragraph_group),
                        sentences=list(paragraph_group),
                        source_start=paragraph_group[0].char_start,
                        source_end=paragraph_group[-1].char_end,
                    )
                )

        flush()
        # Shift bare enumerators to following draft so no chunk ends with a stranded enumerator
        bare_enum_pattern = r"(?:^|[\s:;])(?:\d{1,2}\.|\([a-z0-9ivx]+\)|(?<![a|p]\.)\b[ivx]{1,4}\.|(?<![a|p]\.)\b[a-z]\.)\s*$"
        for index in range(len(drafts) - 1):
            draft = drafts[index]
            next_draft = drafts[index + 1]
            while len(draft.sentences) > 1 and re.search(bare_enum_pattern, draft.sentences[-1].text.strip(), re.IGNORECASE):
                moved = draft.sentences.pop()
                next_draft.sentences.insert(0, moved)
                next_draft.source_start = moved.char_start
                draft.source_end = draft.sentences[-1].char_end

        drafts = [d for d in drafts if d.sentences]

        for index, draft in enumerate(drafts):
            if draft.source_start is None or draft.source_start < 0:
                continue
            draft.source_end = (
                drafts[index + 1].source_start
                if index + 1 < len(drafts)
                else len(document.normalized_text)
            )
            if draft.source_end is not None and draft.source_start <= draft.source_end:
                draft.src_text = document.normalized_text[draft.source_start : draft.source_end]
        return drafts

    @staticmethod
    def _split_oversized_sentence(
        sentence: Sentence,
        tokenizer: TextTokenizer,
        content_budget: int,
    ) -> List[Sentence]:
        """Split a long legal sentence at semicolons/colons, retaining source offsets."""
        boundaries = [match.end() for match in re.finditer(r"[;:]\s+", sentence.text)]
        if not boundaries:
            return []
        parts = []
        start = 0
        for end in boundaries + [len(sentence.text)]:
            text = sentence.text[start:end]
            if text.strip():
                part = Sentence(
                    sentence_id="{}:part-{:02d}".format(sentence.sentence_id, len(parts) + 1),
                    text=text.strip(),
                    paragraph_id=sentence.paragraph_id,
                    section_id=sentence.section_id,
                    sentence_index=sentence.sentence_index,
                    char_start=sentence.char_start + start,
                    char_end=sentence.char_start + end,
                    enumerator=sentence.enumerator if not parts else None,
                    warnings=sentence.warnings + ["split_long_sentence_at_semicolon_or_colon"],
                )
                if tokenizer.text_token_count(part.text) > content_budget:
                    return []
                parts.append(part)
            start = end
        return parts


class OverlapInjector:
    """Build neighboring context from model tokens without altering srcText."""

    def inject(
        self,
        drafts: Sequence[ChunkDraft],
        tokenizer: TextTokenizer,
        overlap_tokens: int,
        max_tokens: int,
        document_text: Optional[str] = None,
    ) -> List[ChunkWithContext]:
        """Attach whole-sentence source substrings; never decode token IDs into context."""
        result = []
        for index, draft in enumerate(drafts):
            preceding_context = self._whole_sentence_context(
                drafts[index - 1] if index else None,
                tokenizer,
                overlap_tokens,
                document_text,
                preceding=True,
            )
            following_context = self._whole_sentence_context(
                drafts[index + 1] if index + 1 < len(drafts) else None,
                tokenizer,
                overlap_tokens,
                document_text,
                preceding=False,
            )
            while self._combined_count(tokenizer, preceding_context, draft.src_text, following_context) > max_tokens:
                if preceding_context is not None:
                    preceding_context = None
                elif following_context is not None:
                    following_context = None
                else:
                    raise ValueError("Chunk source text exceeds max_tokens after context removal")

            content_count = self._content_count(tokenizer, draft.src_text)
            context_count = self._content_count(
                tokenizer,
                " ".join(context for context in (preceding_context, following_context) if context),
            )
            result.append(ChunkWithContext(draft, preceding_context, following_context, content_count, context_count))
        return result

    @classmethod
    def _whole_sentence_context(
        cls,
        draft: Optional[ChunkDraft],
        tokenizer: TextTokenizer,
        token_budget: int,
        document_text: Optional[str],
        preceding: bool,
    ) -> Optional[str]:
        if draft is None:
            return None
        if draft.sentences and document_text is not None:
            selected = []
            ordered = list(reversed(draft.sentences)) if preceding else list(draft.sentences)
            for sentence in ordered:
                selected.append(sentence)
                first = selected[-1] if preceding else selected[0]
                last = selected[0] if preceding else selected[-1]
                candidate = document_text[first.char_start:last.char_end]
                if len(selected) > 1 and cls._content_count(tokenizer, candidate) > token_budget:
                    selected.pop()
                    break
                if len(selected) == 1 and cls._content_count(tokenizer, candidate) > max(token_budget, 55):
                    selected.pop()
                    break
            if not selected:
                return None
            first = selected[-1] if preceding else selected[0]
            last = selected[0] if preceding else selected[-1]
            start, end = first.char_start, last.char_end
            candidate = document_text[start:end]
            if candidate not in draft.src_text:
                return None
            return candidate

        # Compatibility path for custom Draft callers without source offsets.
        words = draft.src_text.split()
        selected_words = []
        ordered_words = list(reversed(words)) if preceding else words
        for word in ordered_words:
            selected_words.append(word)
            candidate = " ".join(reversed(selected_words)) if preceding else " ".join(selected_words)
            if len(selected_words) > 1 and cls._content_count(tokenizer, candidate) > token_budget:
                selected_words.pop()
                break
        if not selected_words:
            return None
        candidate = " ".join(reversed(selected_words)) if preceding else " ".join(selected_words)
        return candidate if candidate in draft.src_text else None

    @staticmethod
    def _content_count(tokenizer: TextTokenizer, text: str) -> int:
        if not text:
            return 0
        measure = getattr(tokenizer, "measure", None)
        if measure:
            return measure(text)[0]
        counter = getattr(tokenizer, "text_token_count", None)
        return counter(text) if counter else len(tokenizer.encode(text))

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

    def build(
        self,
        document_id: str,
        chunks: Sequence[ChunkWithContext],
        tokenizer: TextTokenizer,
        max_tokens: int,
    ) -> List[TranslationChunk]:
        """Assign IDs and workflow defaults only after chunk boundaries are final."""
        built = []
        for index, chunk in enumerate(chunks):
            measure = getattr(tokenizer, "measure", None)
            if measure:
                content_token_count, unknown_rate = measure(chunk.draft.src_text)
                context_token_count = sum(
                    measure(context)[0]
                    for context in (chunk.preceding_context, chunk.following_context)
                    if context
                )
            else:
                content_token_count = len(tokenizer.encode(chunk.draft.src_text))
                context_token_count = sum(
                    len(tokenizer.encode(context))
                    for context in (chunk.preceding_context, chunk.following_context)
                    if context
                )
                unknown_rate = 0.0
            warnings = [
                warning
                for sentence in chunk.draft.sentences
                for warning in sentence.warnings
            ]
            if unknown_rate > 0:
                warnings.append("source_contains_unknown_tokens")
            if re.search(r"(?:\b\d{1,2}\.|\([ivxlcdm]+\))\s*$", chunk.draft.src_text, re.IGNORECASE):
                warnings.append("chunk_ends_with_stranded_enumerator")
            if content_token_count + context_token_count > max_tokens:
                warnings.append("chunk_and_context_exceed_token_budget")
            chunk_id = str(uuid.uuid4())
            for sentence in chunk.draft.sentences:
                sentence.chunk_id = chunk_id
            built.append(
                TranslationChunk(
                    chunkId=chunk_id,
                    documentId=document_id,
                    sectionId=chunk.draft.section_id,
                    paragraphId=chunk.draft.paragraph_id,
                    unitIndex=index,
                    srcText=chunk.draft.src_text,
                    precedingContext=chunk.preceding_context,
                    followingContext=chunk.following_context,
                    status="pending",
                    repairCount=0,
                    contentTokenCount=content_token_count,
                    contextTokenCount=context_token_count,
                    unknownTokenRate=unknown_rate,
                    warnings=list(dict.fromkeys(warnings)),
                    sentences=list(chunk.draft.sentences),
                )
            )
        return built


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
        preprocessed.normalized_text,
    )
    chunks = ChunkBuilder().build(
        preprocessed.document.documentId,
        contextualized,
        active_tokenizer,
        budget.max_tokens,
    )
    logger.info(
        "Chunking completed in %.3fs: %d chunks (content budget %d/%d tokens)",
        perf_counter() - started,
        len(chunks),
        budget.content_budget,
        budget.max_tokens,
    )
    return chunks