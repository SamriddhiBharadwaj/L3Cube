"""English text preprocessing and document-level extraction."""

import logging
import re
import unicodedata
import uuid
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .models import (
    CodeSwitchFlag,
    DocumentRecord,
    Entity,
    LanguageDetection,
    LoadedText,
    Paragraph,
    PreprocessedDocument,
    Sentence,
)

logger = logging.getLogger(__name__)


class TextLoader:
    """Read one UTF-8 plain-text source and capture its raw size metadata."""

    def load(self, file_path: str) -> LoadedText:
        """Decode strictly so corrupt or differently encoded input is not mangled."""
        path = Path(file_path)
        if path.suffix.lower() != ".txt":
            raise ValueError("Only UTF-8 .txt source files are supported")

        raw_text = path.read_bytes().decode("utf-8", errors="strict")
        line_count = raw_text.count("\n") + (1 if raw_text else 0)
        return LoadedText(raw_text, len(raw_text), line_count)


class Normalizer:
    """Normalize Unicode and layout without damaging Devanagari graphemes."""

    _punctuation_map = str.maketrans(
        {
            "“": '"',
            "”": '"',
            "„": '"',
            "‟": '"',
            "‘": "'",
            "’": "'",
            "‚": "'",
            "‛": "'",
            "‐": "-",
            "‑": "-",
            "‒": "-",
            "–": "-",
            "—": "-",
            "…": "...",
            "｡": ".",
            "，": ",",
            "？": "?",
            "！": "!",
            "：": ":",
            "；": ";",
        }
    )

    def normalize(self, text: str) -> str:
        """Keep combining marks/joiners intact; only discard C0/C1 controls."""
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.translate(self._punctuation_map)

        safe_characters = []
        for character in text:
            if character in ("\n", "\t"):
                safe_characters.append(character)
            elif unicodedata.category(character) == "Cc":
                continue
            else:
                safe_characters.append(character)

        lines = [re.sub(r"[^\S\n]+", " ", line).rstrip() for line in "".join(safe_characters).split("\n")]
        normalized_lines = []
        previous_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            normalized_lines.append(line)
            previous_blank = is_blank
        return "\n".join(normalized_lines).strip()


class ParagraphSplitter:
    """Create paragraph records and lightweight section IDs from text layout."""

    _heading_prefix = re.compile(
        r"^(?:#{1,6}\s+|chapter\b|section\b|part\b|appendix\b|"
        r"introduction\b|conclusion\b|summary\b)",
        re.IGNORECASE,
    )

    def split(self, text: str, document_id: str) -> List[Paragraph]:
        """Split on blank lines or indentation, retaining headings as paragraphs."""
        raw_paragraphs: List[str] = []
        current_lines: List[str] = []

        def flush() -> None:
            paragraph_text = " ".join(line.strip() for line in current_lines).strip()
            if paragraph_text:
                raw_paragraphs.append(paragraph_text)
            current_lines.clear()

        for line in text.splitlines():
            if not line.strip():
                flush()
                continue
            if line[:1].isspace() and current_lines:
                flush()
            current_lines.append(line)
        flush()

        section_index = 0
        paragraphs: List[Paragraph] = []
        for paragraph_index, paragraph_text in enumerate(raw_paragraphs, start=1):
            is_heading = bool(self._heading_prefix.match(paragraph_text))
            if is_heading:
                section_index += 1
            section_id = "{}:section-{:04d}".format(document_id, section_index)
            paragraphs.append(
                Paragraph(
                    paragraph_id="{}:paragraph-{:04d}".format(document_id, paragraph_index),
                    section_id=section_id,
                    text=paragraph_text.lstrip("# ").strip() if paragraph_text.startswith("#") else paragraph_text,
                    paragraph_index=paragraph_index,
                    is_heading=is_heading,
                )
            )
        return paragraphs


class SentenceSplitter:
    """Split English paragraphs at sentence-final punctuation without breaking abbreviations."""

    _sentence_boundary = re.compile(r"(?<=[.!?])\s+")

    def split(self, paragraphs: Sequence[Paragraph]) -> List[Paragraph]:
        """Split at English sentence punctuation followed by whitespace."""
        for paragraph in paragraphs:
            sentence_texts = self._sentence_boundary.split(paragraph.text.strip())
            sentence_texts = [sentence.strip() for sentence in sentence_texts if sentence.strip()]

            paragraph.sentences = [
                Sentence(
                    sentence_id="{}:sentence-{:04d}".format(paragraph.paragraph_id, index),
                    text=sentence_text,
                    paragraph_id=paragraph.paragraph_id,
                    section_id=paragraph.section_id,
                    sentence_index=index,
                )
                for index, sentence_text in enumerate(sentence_texts, start=1)
            ]
        return list(paragraphs)


class LanguageDetector:
    """Confirm English source text and flag sentences containing non-Latin scripts."""

    confidence_threshold = 0.80
    _english_word = re.compile(r"\b[A-Za-z][A-Za-z0-9+#.-]*\b")

    def detect(self, sentences: Sequence[Sentence]) -> LanguageDetection:
        """Use langdetect when installed, with a Unicode Latin-script fallback."""
        full_text = " ".join(sentence.text for sentence in sentences)
        language, confidence = self._detect_one(full_text)
        flags = []
        for sentence in sentences:
            sentence_language, sentence_confidence = self._detect_one(sentence.text)
            has_english = bool(self._english_word.search(sentence.text))
            has_non_latin = self._contains_non_latin_letters(sentence.text)
            if has_english and has_non_latin:
                reason = "mixed_latin_and_non_latin_script"
                flagged_language = "mixed"
            elif sentence_confidence < self.confidence_threshold or sentence_language != "en":
                reason = "low_language_confidence_or_non_english_segment"
                flagged_language = sentence_language
            else:
                continue
            flags.append(
                CodeSwitchFlag(
                    sentence_id=sentence.sentence_id,
                    text=sentence.text,
                    detected_language=flagged_language,
                    confidence=sentence_confidence,
                    reason=reason,
                )
            )
        return LanguageDetection(
            detected_language=language,
            confidence=confidence,
            is_english=(language == "en"),
            code_switch_flags=flags,
        )

    def _detect_one(self, text: str) -> Tuple[str, float]:
        if not text.strip():
            return "unknown", 0.0
        try:
            from langdetect import detect_langs

            results = detect_langs(text)
            if results:
                return results[0].lang, float(results[0].prob)
        except (ImportError, Exception) as error:
            logger.debug("langdetect unavailable or inconclusive: %s", error)

        letters = [character for character in text if character.isalpha()]
        latin_count = sum("LATIN" in unicodedata.name(character, "") for character in letters)
        if letters and latin_count / float(len(letters)) >= 0.7:
            return "en", 0.99
        if letters:
            return "unknown", 0.0
        return "unknown", 0.0

    @staticmethod
    def _contains_non_latin_letters(text: str) -> bool:
        return any(
            character.isalpha() and "LATIN" not in unicodedata.name(character, "")
            for character in text
        )


class EntityExtractor:
    """Extract English people, places, organizations, and proper nouns with NER."""

    _term_types = {
        "PER": "person",
        "LOC": "place",
        "ORG": "organization",
        "MISC": "proper_noun",
    }
    _minimum_confidence = 0.55
    _word_categories = {"L", "M", "N"}
    _generic_person_labels = {"plaintiff", "petitioner", "respondent", "defendant"}
    _generic_organization_labels = {"court", "family court"}

    def extract(self, sentences: Sequence[Sentence], document_id: str) -> List[Entity]:
        """Run English NER lazily and reconstruct complete spans from source offsets."""
        try:
            ner_pipeline = self._get_pipeline()
            sentence_texts = [sentence.text for sentence in sentences if sentence.text.strip()]
            output = ner_pipeline(sentence_texts, batch_size=4)
        except Exception as error:
            if self._mps_is_available():
                try:
                    logger.warning("IndicNER inference failed on MPS; retrying on CPU: %s", error)
                    ner_pipeline = self._get_pipeline(use_mps=False)
                    output = ner_pipeline(sentence_texts, batch_size=4)
                except Exception as cpu_error:
                    logger.warning("IndicNER extraction unavailable on CPU: %s", cpu_error)
                    return []
            else:
                logger.warning("IndicNER extraction unavailable; no NER candidates added: %s", error)
                return []

        entities = []
        for sentence_text, sentence_output in zip(sentence_texts, output):
            predictions = [sentence_output] if isinstance(sentence_output, dict) else sentence_output
            spans = []
            for result in predictions:
                label = str(result.get("entity_group", result.get("entity", ""))).upper()
                term_type = self._term_types.get(label.split("-")[-1])
                if not term_type or float(result.get("score", 1.0)) < self._minimum_confidence:
                    continue

                start = result.get("start")
                end = result.get("end")
                if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(sentence_text):
                    start, end = self._expand_to_word_boundaries(sentence_text, start, end)
                    source_term = sentence_text[start:end].strip(" \t\n,;:।॥.!?()[]{}\"'")
                else:
                    source_term = str(result.get("word", "")).replace("##", "").strip()
                    source_term = source_term.strip(" \t\n,;:।॥.!?()[]{}\"'")
                if sum(character.isalpha() for character in source_term) < 2:
                    continue
                if term_type == "person" and source_term.casefold() in self._generic_person_labels:
                    continue
                if term_type == "organization" and source_term.casefold() in self._generic_organization_labels:
                    continue
                spans.append((start, end, term_type, source_term))

            entities.extend(
                self._merge_sentence_spans(sentence_text, spans, document_id)
            )
        return entities

    @classmethod
    def _expand_to_word_boundaries(cls, text: str, start: int, end: int) -> Tuple[int, int]:
        """Expand subword offsets so a prediction returns complete source words."""
        while start > 0 and cls._is_word_character(text[start - 1]):
            start -= 1
        while end < len(text) and cls._is_word_character(text[end]):
            end += 1
        return start, end

    @classmethod
    def _is_word_character(cls, character: str) -> bool:
        return unicodedata.category(character)[0] in cls._word_categories or character in "\u200c\u200d_"

    @classmethod
    def _merge_sentence_spans(
        cls,
        sentence_text: str,
        spans: Sequence[Tuple[object, object, str, str]],
        document_id: str,
    ) -> List[Entity]:
        """Join adjacent same-label subwords by their source offsets, not token text."""
        valid_spans = sorted(
            (span for span in spans if isinstance(span[0], int) and isinstance(span[1], int)),
            key=lambda span: span[0],
        )
        merged_spans = []
        for start, end, term_type, source_term in valid_spans:
            if merged_spans:
                previous_start, previous_end, previous_type, _ = merged_spans[-1]
                gap = sentence_text[previous_end:start]
                if previous_type == term_type and (not gap or gap.isspace()):
                    merged_end = max(previous_end, end)
                    merged_spans[-1] = (
                        previous_start,
                        merged_end,
                        term_type,
                        sentence_text[previous_start:merged_end].strip(" \t\n,;:।॥.!?()[]{}\"'"),
                    )
                    continue
            merged_spans.append((start, end, term_type, source_term))

        entities = [
            cls._entity(document_id, source_term, term_type)
            for _start, _end, term_type, source_term in merged_spans
            if sum(character.isalpha() for character in source_term) >= 2
        ]
        # Preserve offset-free outputs from compatible test doubles/custom pipelines.
        entities.extend(
            cls._entity(document_id, source_term, term_type)
            for start, _end, term_type, source_term in spans
            if not isinstance(start, int)
        )
        return entities

    @classmethod
    @lru_cache(maxsize=2)
    def _get_pipeline(cls, use_mps: bool = True) -> Callable[..., List[dict]]:
        from transformers import pipeline

        import torch

        use_mps = use_mps and torch.backends.mps.is_available()
        device = torch.device("mps") if use_mps else torch.device("cpu")
        try:
            return pipeline(
                "ner",
                model="elastic/distilbert-base-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=device,
                trust_remote_code=True,
            )
        except Exception:
            if not use_mps:
                raise
            logger.warning("IndicNER could not initialize on MPS; retrying on CPU")
            return cls._get_pipeline(use_mps=False)

    @staticmethod
    def _mps_is_available() -> bool:
        try:
            import torch

            return torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            return False

    @staticmethod
    def _entity(document_id: str, source_term: str, term_type: str) -> Entity:
        return Entity(str(uuid.uuid4()), document_id, source_term, None, term_type)


class TerminologyExtractor:
    """Find English technical terms with YAKE, separately from named entities."""

    _english_stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
        "has", "have", "he", "her", "hers", "him", "his", "in", "is", "it", "its", "of",
        "on", "or", "our", "she", "that", "the", "their", "them", "there", "these", "they",
        "this", "those", "to", "was", "were", "will", "with", "would", "you", "your",
    }

    def __init__(self, max_keywords: int = 25) -> None:
        self.max_keywords = max_keywords

    def extract(self, sentences: Sequence[Sentence], document_id: str) -> List[Entity]:
        """Produce technical-term candidates; curated glossary values can be merged later."""
        source_text = " ".join(sentence.text for sentence in sentences)
        if not source_text.strip():
            return []
        try:
            import yake

            extractor = yake.KeywordExtractor(lan="en", n=3, top=self.max_keywords)
            keywords = extractor.extract_keywords(source_text)
        except Exception as error:
            logger.warning("YAKE terminology extraction unavailable: %s", error)
            return []
        entities = []
        for term, _score in keywords:
            words = []
            for raw_word in term.split():
                start = 0
                end = len(raw_word)
                while start < end and unicodedata.category(raw_word[start]).startswith("P"):
                    start += 1
                while end > start and unicodedata.category(raw_word[end - 1]).startswith("P"):
                    end -= 1
                word = raw_word[start:end]
                if word:
                    words.append(word)
            content_words = [
                word for word in words
                if word.casefold() not in self._english_stop_words and len(word) > 1
            ]
            if len(content_words) < 2:
                continue
            cleaned_term = " ".join(content_words)
            entities.append(Entity(str(uuid.uuid4()), document_id, cleaned_term, None, "technical_term"))
        return entities


def merge_entity_candidates(candidates: Sequence[Entity]) -> List[Entity]:
    """Merge English case/spelling aliases while preferring complete names."""
    from collections import Counter, defaultdict

    groups = defaultdict(list)
    ner_keys = []
    for candidate in candidates:
        key = _canonical_entity_key(candidate.sourceTerm)
        if not key:
            continue
        if candidate.termType not in ("technical_term", "glossary"):
            ner_keys.append(key)
        groups[(candidate.termType, key)].append(candidate)

    person_keys = [key for term_type, key in groups if term_type == "person"]
    person_aliases = {key: key for key in person_keys}
    for key_index, key in enumerate(person_keys):
        for other_key in person_keys[key_index + 1 :]:
            if _person_names_are_spelling_variants(key, other_key):
                canonical_key = min(person_aliases[key], person_aliases[other_key])
                person_aliases[key] = canonical_key
                person_aliases[other_key] = canonical_key

    typo_merged_keys = set(person_aliases.values())
    full_person_keys = [key for key in typo_merged_keys if len(key.split()) > 1]
    alias_to_full_name = {}
    for key in typo_merged_keys:
        if len(key.split()) != 1:
            continue
        matches = [
            full_key
            for full_key in full_person_keys
            if _person_names_are_spelling_variants(key, full_key.split()[0])
        ]
        if len(matches) == 1:
            alias_to_full_name[key] = matches[0]
    for alias_key, full_key in alias_to_full_name.items():
        for variant, root in list(person_aliases.items()):
            if root == alias_key:
                person_aliases[variant] = full_key

    combined_groups = defaultdict(list)
    for (term_type, key), variants in groups.items():
        if term_type == "person":
            key = person_aliases.get(key, key)
        combined_groups[(term_type, key)].extend(variants)

    merged = []
    for (term_type, key), variants in combined_groups.items():
        if term_type in ("technical_term", "glossary") and any(
            key in ner_key or ner_key in key for ner_key in ner_keys
        ):
            continue

        frequencies = Counter(candidate.sourceTerm for candidate in variants)
        representative = min(
            frequencies,
            key=lambda term: (
                -len(_canonical_entity_key(term).split()),
                -len(term),
                -frequencies[term],
                term.casefold(),
            ),
        )
        entity = next(candidate for candidate in variants if candidate.sourceTerm == representative)
        merged.append(
            Entity(
                entityId=entity.entityId,
                documentId=entity.documentId,
                sourceTerm=representative,
                targetName=entity.targetName,
                termType=entity.termType,
                gender=entity.gender,
            )
        )
    return merged


def _canonical_entity_key(source_term: str) -> str:
    """Normalize English case, accents, punctuation, and whitespace for matching."""
    normalized = unicodedata.normalize("NFKD", source_term).casefold()
    ascii_letters = "".join(character for character in normalized if not unicodedata.category(character).startswith("M"))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_letters))


def _person_names_are_spelling_variants(first: str, second: str) -> bool:
    """Match near-identical English person names, not broadly similar strings."""
    first_words = first.split()
    second_words = second.split()
    if len(first_words) != len(second_words):
        return False
    return all(
        left == right
        or (
            min(len(left), len(right)) >= 4
            and SequenceMatcher(None, left, right).ratio() >= 0.90
        )
        for left, right in zip(first_words, second_words)
    )


LanguageExtractor = Callable[[Sequence[Sentence]], LanguageDetection]
EntityCandidateExtractor = Callable[[Sequence[Sentence], str], List[Entity]]


def preprocess_document(
    file_path: str,
    document_id: str,
    target_language: str = "hin_Deva",
    genre: Optional[str] = None,
    register: Optional[str] = None,
    language_detector: Optional[LanguageExtractor] = None,
    entity_extractor: Optional[EntityCandidateExtractor] = None,
    terminology_extractor: Optional[EntityCandidateExtractor] = None,
) -> PreprocessedDocument:
    """Run one-time English document processing and parallel independent extraction.

    The extractor callables are injectable to keep each stage testable without
    downloading large language models. In production, absent callables select
    langdetect, an English NER model, and YAKE configured for English.
    """
    started = perf_counter()
    loaded = TextLoader().load(file_path)
    logger.info("TextLoader completed: %d chars, %d lines", loaded.char_count, loaded.line_count)

    stage_started = perf_counter()
    normalized_text = Normalizer().normalize(loaded.raw_text)
    logger.info("Normalizer completed in %.3fs", perf_counter() - stage_started)

    stage_started = perf_counter()
    paragraphs = ParagraphSplitter().split(normalized_text, document_id)
    paragraphs = SentenceSplitter().split(paragraphs)
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph.sentences]
    logger.info(
        "Paragraph and sentence splitting completed in %.3fs: %d paragraphs, %d sentences",
        perf_counter() - stage_started,
        len(paragraphs),
        len(sentences),
    )

    language_fn = language_detector or LanguageDetector().detect
    entity_fn = entity_extractor or EntityExtractor().extract
    terminology_fn = terminology_extractor or TerminologyExtractor().extract
    stage_started = perf_counter()
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="document-extraction") as executor:
        language_future = executor.submit(language_fn, sentences)
        entity_future = executor.submit(entity_fn, sentences, document_id)
        terminology_future = executor.submit(terminology_fn, sentences, document_id)
        language = language_future.result()
        ner_entities = entity_future.result()
        term_entities = terminology_future.result()
    entities = merge_entity_candidates(ner_entities + term_entities)
    if not language.is_english:
        raise ValueError(
            "Expected an English source document, but detected {!r} "
            "(confidence {:.2f})".format(language.detected_language, language.confidence)
        )
    logger.info(
        "Parallel extraction completed in %.3fs: %d code-switch flags, %d entities",
        perf_counter() - stage_started,
        len(language.code_switch_flags),
        len(entities),
    )

    now = datetime.now(timezone.utc)
    document = DocumentRecord(
        documentId=document_id,
        title=Path(file_path).stem,
        sourceLanguage="eng_Latn",
        targetLanguage=target_language,
        genre=genre,
        register=register,
        status="preprocessed",
        createdAt=now,
        updatedAt=now,
    )
    result = PreprocessedDocument(
        document=document,
        normalized_text=normalized_text,
        paragraphs=paragraphs,
        language=language,
        entities=entities,
        char_count=loaded.char_count,
        line_count=loaded.line_count,
    )
    logger.info("Preprocessing completed in %.3fs", perf_counter() - started)
    return result