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
    PageHeader,
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
    """Normalize extracted English text and recover layout cues from PDF text."""

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
    _page_header = re.compile(
        r"CM\s*\(\s*M\s*\)\s*47/2022\s*&\s*CM\s*\(\s*M\s*\)\s*211/2022"
        r"\s*Page\s+(\d+)\s+of\s+(\d+)",
        re.IGNORECASE,
    )
    _line_enumerator = re.compile(
        r"^(?:\d{1,2}\.|\([ivxlcdm]+\)|\([a-z]\)|[ivxlcdm]+\.|\b[a-z]\.)\s+",
        re.IGNORECASE,
    )

    def normalize(self, text: str) -> str:
        """Return normalized text, while preserving meaningful legal list boundaries."""
        return self.normalize_with_metadata(text)[0]

    def normalize_with_metadata(self, text: str) -> Tuple[str, List[PageHeader]]:
        """Repair PDF extraction artifacts and retain removed page labels as metadata."""
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Repair letter + "s / ‟s / ”s -> 's
        text = re.sub(r'([A-Za-z])["‟”\']s\b', r"\1's", text)
        text = text.translate(self._punctuation_map)
        text = re.sub(r'([A-Za-z])"s\b', r"\1's", text)
        text = re.sub(r"\b(Mr|Mrs|Ms|Miss|Dr|Sh|Smt)\.(?=[A-Za-z])", r"\1. ", text)
        text = re.sub(r"\bwell-\s+reasoned\b", "well-reasoned", text, flags=re.IGNORECASE)

        safe_characters = []
        for character in text:
            if character in ("\n", "\t"):
                safe_characters.append(character)
            elif unicodedata.category(character) == "Cc":
                continue
            else:
                safe_characters.append(character)

        cleaned = "".join(safe_characters)

        # Extract and remove multiline page headers cleanly so no blank line is left mid-sentence
        multiline_header = re.compile(
            r"^[ \t]*(.*?)\bPage\s+(\d+)\s+of\s+(\d+)[ \t]*(?:\n|\Z)",
            re.MULTILINE | re.IGNORECASE,
        )
        page_headers = [
            PageHeader(int(match.group(2)), int(match.group(3)), match.group(0).strip())
            for match in multiline_header.finditer(cleaned)
        ]
        cleaned = multiline_header.sub("", cleaned)

        # Flat header pattern for single-line PDF input
        flat_header = re.compile(
            r"(?:CM\s*\(\s*M\s*\)\s*\d+/\d+\s*(?:&|,|\band\b)?\s*)*\s*Page\s+(\d+)\s+of\s+(\d+)",
            re.IGNORECASE,
        )
        if not page_headers:
            for match in flat_header.finditer(cleaned):
                page_headers.append(
                    PageHeader(int(match.group(1)), int(match.group(2)), match.group(0).strip())
                )
        cleaned = flat_header.sub(" ", cleaned)

        # Structure recovery fallback if input has no newlines
        if cleaned.count("\n") < 3:
            cleaned = re.sub(
                r'(?<=[.!?"])\s+(?=(?:\d{1,2}\.|\([ivxlcdm]+\)|\([a-z]\)|[ivxlcdm]+\.|\b[a-z]\.)\s+[A-Za-z])',
                "\n\n",
                cleaned,
            )

        normalized_parts: List[str] = []
        for line in cleaned.split("\n"):
            if not line.strip():
                if normalized_parts and normalized_parts[-1] != "\n\n":
                    normalized_parts.append("\n\n")
                continue

            is_indented_paragraph = bool(line[:1].isspace())
            stripped = re.sub(r"\s+", " ", line).strip()
            is_enumerator = bool(self._line_enumerator.match(stripped))
            if normalized_parts:
                if normalized_parts[-1] == "\n\n":
                    pass
                elif is_indented_paragraph or is_enumerator:
                    normalized_parts.append("\n\n")
                else:
                    normalized_parts.append(" ")
            normalized_parts.append(stripped)

        normalized_text = "".join(normalized_parts).strip()
        normalized_text = re.sub(r"[ \t]+\n", "\n", normalized_text)
        return normalized_text, page_headers


class ParagraphSplitter:
    """Create paragraph records and lightweight section IDs from text layout."""

    _heading_prefix = re.compile(
        r"^(?:#{1,6}\s+|chapter\b|section\b|part\b|appendix\b|"
        r"introduction\b|conclusion\b|summary\b)",
        re.IGNORECASE,
    )

    def split(self, text: str, document_id: str) -> List[Paragraph]:
        """Split blank-line and recovered-enumerator boundaries with source offsets."""
        section_index = 0
        expected_item_num = 1
        paragraphs: List[Paragraph] = []
        blocks = re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text, flags=re.DOTALL)
        for paragraph_index, match in enumerate(blocks, start=1):
            paragraph_text = match.group()
            is_heading = bool(self._heading_prefix.match(paragraph_text))
            enumerator_match = Normalizer._line_enumerator.match(paragraph_text)
            top_level_num_match = re.match(r"^(\d{1,2})\.\s+", paragraph_text)
            if is_heading:
                section_index += 1
            elif top_level_num_match:
                item_num = int(top_level_num_match.group(1))
                if item_num == expected_item_num or (section_index == 0 and item_num == 1):
                    section_index += 1
                    expected_item_num = item_num + 1
                elif item_num > expected_item_num + 3:
                    # Cited block numbering (e.g. 33., 34. inside section 9)
                    pass
                else:
                    section_index += 1
                    expected_item_num = item_num + 1
            section_id = "{}:section-{:04d}".format(document_id, max(1, section_index))
            paragraphs.append(
                Paragraph(
                    paragraph_id="{}:paragraph-{:04d}".format(document_id, paragraph_index),
                    section_id=section_id,
                    text=paragraph_text,
                    paragraph_index=paragraph_index,
                    is_heading=is_heading,
                    char_start=match.start(),
                    char_end=match.end(),
                    enumerator=enumerator_match.group().strip() if enumerator_match else None,
                )
            )
        return paragraphs


class SentenceSplitter:
    """Split English text while retaining abbreviation, list, and source-offset data."""

    _abbreviations = re.compile(
        r"\b(?:a\.m\.|p\.m\.|mr\.|mrs\.|ms\.|dr\.|sh\.|smt\.|adv\.|hon\.|no\.|nos\.|vs\.|v\.|"
        r"vol\.|sec\.|art\.|para\.|e\.g\.|i\.e\.|etc\.|u\.s\.|u\.k\.|del\.|ltd\.|pvt\.|co\.|corp\.)",
        re.IGNORECASE,
    )
    _enumerator = re.compile(
        r"^\s*(\d{1,2}\.|\([ivxlcdm]+\)|\([a-z]\)|[ivxlcdm]+\.|\b[a-z]\.)\s+",
        re.IGNORECASE,
    )
    _bare_enumerator = re.compile(
        r"^\s*(?:\d{1,2}\.|\([ivxlcdm]+\)|\([a-z]\)|[ivxlcdm]+\.|\b[a-z]\.)\s*$",
        re.IGNORECASE,
    )

    def split(self, paragraphs: Sequence[Paragraph]) -> List[Paragraph]:
        """Split at sentence punctuation, never at decimal/abbreviation or list markers."""
        for paragraph in paragraphs:
            sentence_ranges = self._split_ranges(paragraph.text)
            sentences = []
            for index, (start, end) in enumerate(sentence_ranges, start=1):
                sentence_text = paragraph.text[start:end]
                enumerator_match = self._enumerator.match(sentence_text)
                sentences.append(
                    Sentence(
                        sentence_id="{}:sentence-{:04d}".format(paragraph.paragraph_id, index),
                        text=sentence_text,
                        paragraph_id=paragraph.paragraph_id,
                        section_id=paragraph.section_id,
                        sentence_index=index,
                        char_start=paragraph.char_start + start,
                        char_end=paragraph.char_start + end,
                        enumerator=enumerator_match.group(1) if enumerator_match else None,
                    )
                )
            paragraph.sentences = self._attach_stranded_enumerators(sentences)

        # Cross-paragraph check: if a paragraph ends with a bare enumerator, attach to next paragraph's first sentence
        for p_idx in range(len(paragraphs) - 1):
            curr_p = paragraphs[p_idx]
            next_p = paragraphs[p_idx + 1]
            if curr_p.sentences and self._bare_enumerator.match(curr_p.sentences[-1].text):
                bare = curr_p.sentences.pop()
                if next_p.sentences:
                    first = next_p.sentences[0]
                    first.text = bare.text.strip() + " " + first.text.lstrip()
                    first.char_start = bare.char_start
                    first.enumerator = bare.text.strip()
                else:
                    next_p.sentences.append(bare)

        return list(paragraphs)

    def _split_ranges(self, text: str) -> List[Tuple[int, int]]:
        protected_periods = {
            position
            for match in self._abbreviations.finditer(text)
            for position in range(match.start(), match.end())
            if text[position] == "."
        }
        boundaries = []
        start = 0
        index = 0
        while index < len(text):
            character = text[index]
            if character not in ".!?":
                index += 1
                continue
            numeric_marker = re.match(r"^\s*(\d{1,2}\.)\s+", text[start:])
            is_enumerator_period = bool(
                numeric_marker
                and index == start + numeric_marker.start(1) + len(numeric_marker.group(1)) - 1
            )
            if character == "." and (
                index in protected_periods
                or (index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit())
                or is_enumerator_period
            ):
                index += 1
                continue

            end = index + 1
            while end < len(text) and text[end] in '"\'”’)]}':
                end += 1
            next_start = end
            while next_start < len(text) and text[next_start].isspace():
                next_start += 1
            if next_start == end:
                index += 1
                continue
            if self._enumerator.match(text[next_start:]):
                boundaries.append((start, end))
                start = next_start
                index = next_start
                continue
            next_letter = re.search(r"[A-Za-z]", text[next_start:])
            if next_letter and text[next_start + next_letter.start()].islower():
                index += 1
                continue
            boundaries.append((start, end))
            start = next_start
            index = next_start

        if start < len(text):
            boundaries.append((start, len(text)))
        return [(start, end) for start, end in boundaries if text[start:end].strip()]

    @classmethod
    def _attach_stranded_enumerators(cls, sentences: List[Sentence]) -> List[Sentence]:
        """Join a marker-only fragment to its following sentence if malformed input creates one."""
        result = []
        index = 0
        while index < len(sentences):
            sentence = sentences[index]
            if cls._bare_enumerator.match(sentence.text) and index + 1 < len(sentences):
                following = sentences[index + 1]
                marker = sentence.text.strip()
                following.text = marker + " " + following.text.lstrip()
                following.char_start = sentence.char_start
                following.enumerator = marker
                index += 1
            else:
                result.append(sentence)
            index += 1
        for sentence_index, sentence in enumerate(result, start=1):
            sentence.sentence_index = sentence_index
            sentence.sentence_id = "{}:sentence-{:04d}".format(sentence.paragraph_id, sentence_index)
        return result


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
    _generic_person_labels = {"plaintiff", "petitioner", "respondent", "defendant", "appellant", "applicant"}
    _generic_organization_labels = {"court", "family court", "judge family", "cm (m", "sma", "lis", "del"}
    _honorifics = {
        "mr", "mrs", "ms", "miss", "dr", "sh", "shri", "smt", "adv", "hon",
        "honorable", "honourable", "justice", "counsel", "senior counsel",
    }
    _demonym_labels = {"canadian", "american", "british", "indian", "australian", "german", "french", "chinese"}
    _identifier_patterns = (
        ("case_number", re.compile(r"\bCM\s*\(M\)\s*(?:No\.?\s*)?\d{1,4}/\d{4}\b", re.IGNORECASE)),
        ("case_number", re.compile(r"\bSMA\s+No\.?\s*\d{1,4}/\d{4}\b", re.IGNORECASE)),
        ("case_number", re.compile(r"\bGuardianship\s+Petition\s+(?:No|no)\.?\s*\d{1,4}/\d{4}\b", re.IGNORECASE)),
        ("case_number", re.compile(r"\b(?:Petition\s+)?No\.?\s*\d{1,4}/\d{4}\b", re.IGNORECASE)),
        ("citation", re.compile(r"\b(?:\d{4}\s+)?SCC\s+OnLine\s+Del(?:\s+\d{1,6})?\b", re.IGNORECASE)),
        ("identifier", re.compile(r"\b(?:GA|AM)\d{4,}\b", re.IGNORECASE)),
        ("date", re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")),
        ("amount", re.compile(r"\b(?:USD|INR|Rs\.?|\$|₹)\s?[\d,]+(?:\.\d{1,2})?\b", re.IGNORECASE)),
    )

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
        sentence_ids = [sentence.sentence_id for sentence in sentences if sentence.text.strip()]
        for sentence_text, sentence_id, sentence_output in zip(sentence_texts, sentence_ids, output):
            predictions = [sentence_output] if isinstance(sentence_output, dict) else sentence_output
            spans = []
            for identifier_type, pattern in self._identifier_patterns:
                for identifier_match in pattern.finditer(sentence_text):
                    entities.append(
                        self._entity(
                            document_id,
                            identifier_match.group(),
                            identifier_type,
                            source="pattern",
                            confidence=1.0,
                            sentence_id=sentence_id,
                        )
                    )
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

                # Strip trailing abbreviations: e.g. "Guardianship Petition No." -> "Guardianship Petition"
                source_term = re.sub(r"\s+\b(?:no\.?|nos\.?|vs\.?|v\.?|del\.?)\s*$", "", source_term, flags=re.IGNORECASE).strip()
                # Clean unbalanced parens
                if source_term.count("(") > source_term.count(")"):
                    source_term = source_term.replace("(", "").strip()
                if source_term.count(")") > source_term.count("("):
                    source_term = source_term.replace(")", "").strip()
                source_term = source_term.strip(" \t\n,;:।॥.!?()[]{}\"'")

                if sum(character.isalpha() for character in source_term) < 2:
                    continue
                if source_term.casefold().rstrip(".") in self._honorifics:
                    continue
                if self._identifier_type(source_term):
                    continue
                if term_type == "proper_noun" and source_term.casefold() in self._demonym_labels:
                    continue
                if term_type == "organization" and self._looks_like_case_header(source_term):
                    continue
                if term_type == "person" and source_term.casefold() in self._generic_person_labels:
                    continue
                if term_type == "organization" and source_term.casefold() in self._generic_organization_labels:
                    continue
                spans.append((start, end, term_type, source_term, float(result.get("score", 1.0))))

            entities.extend(
                self._merge_sentence_spans(sentence_text, spans, document_id, sentence_id)
            )
        return entities

    @staticmethod
    def _looks_like_case_header(value: str) -> bool:
        return bool(re.match(r"^(?:CM\s*\(M|SMA|LIS|SCC\s+OnLine)\b", value, re.IGNORECASE))

    @classmethod
    def _expand_to_word_boundaries(cls, text: str, start: int, end: int) -> Tuple[int, int]:
        """Expand subword offsets so a prediction returns complete source words, without crossing punctuation."""
        while start > 0 and cls._is_word_character(text[start - 1]) and text[start - 1] not in ",;:।॥!?()[]{}\"'":
            start -= 1
        while end < len(text) and cls._is_word_character(text[end]) and text[end] not in ",;:।॥!?()[]{}\"'":
            end += 1
        return start, end

    @classmethod
    def _is_word_character(cls, character: str) -> bool:
        return unicodedata.category(character)[0] in cls._word_categories or character in "\u200c\u200d_"

    @classmethod
    def _merge_sentence_spans(
        cls,
        sentence_text: str,
        spans: Sequence[Tuple[object, object, str, str, float]],
        document_id: str,
        sentence_id: Optional[str] = None,
    ) -> List[Entity]:
        """Join adjacent same-label subwords by their source offsets, not token text."""
        valid_spans = sorted(
            (span for span in spans if isinstance(span[0], int) and isinstance(span[1], int)),
            key=lambda span: span[0],
        )
        merged_spans = []
        for start, end, term_type, source_term, confidence in valid_spans:
            if merged_spans:
                previous_start, previous_end, previous_type, _, previous_confidence = merged_spans[-1]
                gap = sentence_text[previous_end:start]
                if previous_type == term_type and (not gap or gap.isspace()) and not any(ch in ",;:।॥!?()[]{}\"'-" for ch in gap):
                    merged_end = max(previous_end, end)
                    merged_spans[-1] = (
                        previous_start,
                        merged_end,
                        term_type,
                        sentence_text[previous_start:merged_end].strip(" \t\n,;:।॥.!?()[]{}\"'"),
                        (previous_confidence + confidence) / 2.0,
                    )
                    continue
            merged_spans.append((start, end, term_type, source_term, confidence))

        entities = []
        for start, _end, term_type, source_term, confidence in merged_spans:
            source_term = re.sub(r"\s+\b(?:no\.?|nos\.?|vs\.?|v\.?|del\.?)\s*$", "", source_term, flags=re.IGNORECASE).strip()
            source_term = source_term.strip(" \t\n,;:।॥.!?()[]{}\"'")
            if sum(character.isalpha() for character in source_term) < 2:
                continue
            if source_term.casefold().rstrip(".") in cls._honorifics:
                continue
            if source_term.casefold() in cls._generic_person_labels or source_term.casefold() in cls._generic_organization_labels:
                continue
            entities.append(
                cls._entity(
                    document_id,
                    source_term,
                    term_type,
                    gender=cls._infer_gender(sentence_text, start),
                    source="ner",
                    confidence=confidence,
                    sentence_id=sentence_id,
                )
            )
        # Preserve offset-free outputs from compatible test doubles/custom pipelines.
        entities.extend(
            cls._entity(document_id, source_term, term_type)
            for start, _end, term_type, source_term, _confidence in spans
            if not isinstance(start, int)
        )
        return entities

    @classmethod
    def _infer_gender(cls, text: str, start: object) -> Optional[str]:
        if not isinstance(start, int):
            return None
        prefix = text[max(0, start - 20) : start].strip()
        if re.search(r"\b(?:Ms|Mrs|Miss|Smt)\.?\s*$", prefix, re.IGNORECASE):
            return "female"
        if re.search(r"\b(?:Mr|Sh|Shri)\.?\s*$", prefix, re.IGNORECASE):
            return "male"
        return None

    @classmethod
    def _identifier_type(cls, value: str) -> Optional[str]:
        for identifier_type, pattern in cls._identifier_patterns:
            if pattern.fullmatch(value.strip()):
                return identifier_type
        return None

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
    def _entity(
        document_id: str,
        source_term: str,
        term_type: str,
        gender: Optional[str] = None,
        source: str = "ner",
        confidence: Optional[float] = None,
        sentence_id: Optional[str] = None,
    ) -> Entity:
        return Entity(
            str(uuid.uuid4()),
            document_id,
            source_term,
            None,
            term_type,
            gender,
            1,
            sentence_id,
            source,
            confidence,
        )


class TerminologyExtractor:
    """Find English technical terms with YAKE, separately from named entities."""

    _english_stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
        "has", "have", "he", "her", "hers", "him", "his", "in", "is", "it", "its", "of",
        "on", "or", "our", "she", "that", "the", "their", "them", "there", "these", "they",
        "this", "those", "to", "was", "were", "will", "with", "would", "you", "your", "vide",
    }
    _leading_modifiers = {
        "learned", "principal", "honorable", "honourable", "additional",
        "said", "impugned", "such", "inter", "alia", "petitioner",
    }
    _leading_stop = {
        "vide", "decided", "filed", "perused", "held", "stated", "submitted",
        "insofar", "under", "above", "as", "at", "by", "for", "from", "in",
        "into", "of", "on", "or", "to", "with", "and", "but", "the", "a", "an",
    }
    _trailing_non_nouns = {
        "a", "an", "and", "at", "by", "for", "from", "in", "into", "of", "on", "or", "to", "with",
        "decided", "submitted", "stated", "held", "ordered", "filed", "granted", "provided", "vide",
        "passed", "made", "heard", "challenged", "restrained", "quashed", "directed", "observed",
        "agreed", "considered", "noted", "namely", "herein", "therein", "insofar", "under", "above",
        "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "to", "with", "and", "or",
        "also",
    }

    def __init__(self, max_keywords: int = 25, min_mentions: int = 0) -> None:
        self.max_keywords = max_keywords
        self.min_mentions = min_mentions

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
            while content_words and content_words[0].casefold() in self._leading_modifiers:
                content_words.pop(0)
            if len(content_words) < 2:
                continue
            if content_words[0].casefold() in self._leading_stop:
                continue
            if content_words[-1].casefold() in self._trailing_non_nouns:
                continue
            cleaned_term = " ".join(content_words)
            first_sentence_id = next(
                (
                    sentence.sentence_id
                    for sentence in sentences
                    if any(word.casefold() in sentence.text.casefold() for word in content_words)
                ),
                None,
            )
            mention_count = sum(
                1
                for sentence in sentences
                if any(word.casefold() in sentence.text.casefold() for word in content_words)
            )
            if mention_count < self.min_mentions:
                continue
            entities.append(
                Entity(
                    str(uuid.uuid4()),
                    document_id,
                    cleaned_term,
                    None,
                    "technical_term",
                    None,
                    max(1, mention_count),
                    first_sentence_id,
                    "yake",
                    None,
                )
            )
        return entities


def merge_entity_candidates(candidates: Sequence[Entity]) -> List[Entity]:
    """Deduplicate surface aliases first, then vote on their semantic type."""
    from collections import Counter, defaultdict

    records = [(candidate, _canonical_entity_key(candidate.sourceTerm)) for candidate in candidates]
    records = [(candidate, key) for candidate, key in records if key]
    keys = sorted({key for _candidate, key in records})
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    by_key = defaultdict(list)
    for candidate, key in records:
        by_key[key].append(candidate)

    for index, first_key in enumerate(keys):
        first_types = {candidate.termType for candidate in by_key[first_key]}
        for second_key in keys[index + 1 :]:
            second_types = {candidate.termType for candidate in by_key[second_key]}
            if _person_names_are_spelling_variants(first_key, second_key) and (
                "person" in first_types or "person" in second_types
            ):
                union(first_key, second_key)
                continue

            first_words, second_words = first_key.split(), second_key.split()
            short_key, long_key = (
                (first_key, second_key) if len(first_words) <= len(second_words) else (second_key, first_key)
            )
            short_types, long_types = (
                (first_types, second_types) if short_key == first_key else (second_types, first_types)
            )
            short_words, long_words = short_key.split(), long_key.split()
            starts_with_alias = len(short_words) <= 2 and long_words[: len(short_words)] == short_words
            place_alias = "place" in short_types and "place" in long_types
            person_full_name_alias = (
                len(short_words) == 1
                and long_types.intersection({"person", "organization"})
                and "person" in short_types
            )
            case_alias = (
                first_types.intersection({"case_number", "citation"})
                and second_types.intersection({"case_number", "citation"})
                and (short_key in long_key)
            )
            citation_sub = (
                "citation" in long_types
                and short_key in ("scc", "scc online", "scc online del")
            )
            if (starts_with_alias and (place_alias or person_full_name_alias)) or case_alias or citation_sub:
                union(short_key, long_key)

    clusters = defaultdict(list)
    for candidate, key in records:
        clusters[find(key)].append(candidate)

    ner_keys = {
        key for candidate, key in records
        if candidate.source in ("ner", "pattern")
        or (candidate.source == "unknown" and candidate.termType not in ("technical_term", "glossary"))
    }
    merged = []
    for cluster in clusters.values():
        ner_candidates = [
            candidate for candidate in cluster
            if candidate.source in ("ner", "pattern")
            or (candidate.source == "unknown" and candidate.termType not in ("technical_term", "glossary"))
        ]
        term_candidates = [candidate for candidate in cluster if candidate not in ner_candidates]
        if not ner_candidates and term_candidates:
            cluster_keys = [_canonical_entity_key(candidate.sourceTerm) for candidate in term_candidates]
            if any(
                term_key in ner_key or ner_key in term_key
                for term_key in cluster_keys
                for ner_key in ner_keys
            ):
                continue

        voters = ner_candidates or cluster
        type_votes = Counter()
        for candidate in voters:
            source_weight = 3 if candidate.source == "pattern" else 1
            type_votes[candidate.termType] += max(1, candidate.mentionCount) * source_weight
        selected_type = max(
            type_votes,
            key=lambda term_type: (
                type_votes[term_type],
                max((c.confidence or 0.0) for c in voters if c.termType == term_type),
                term_type,
            ),
        )
        selected = [candidate for candidate in voters if candidate.termType == selected_type]
        frequencies = Counter(candidate.sourceTerm for candidate in voters)
        representative = min(
            frequencies,
            key=lambda term: (
                term.isupper(),
                -len(_canonical_entity_key(term).split()),
                -len(term),
                -frequencies[term],
                term.casefold(),
            ),
        )
        representative_record = next(
            (candidate for candidate in voters if candidate.sourceTerm == representative),
            cluster[0],
        )
        confidences = [candidate.confidence for candidate in selected if candidate.confidence is not None]
        sources = {candidate.source for candidate in selected}
        merged.append(
            Entity(
                entityId=representative_record.entityId,
                documentId=representative_record.documentId,
                sourceTerm=representative,
                targetName=next((candidate.targetName for candidate in selected if candidate.targetName), None),
                termType=selected_type,
                gender=_majority_optional([candidate.gender for candidate in cluster]),
                mentionCount=sum(candidate.mentionCount for candidate in selected),
                firstSentenceId=next((candidate.firstSentenceId for candidate in selected if candidate.firstSentenceId), None),
                source="pattern" if "pattern" in sources else ("ner" if ner_candidates else "yake"),
                confidence=sum(confidences) / len(confidences) if confidences else None,
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


def _majority_optional(values: Sequence[Optional[str]]) -> Optional[str]:
    from collections import Counter

    counts = Counter(value for value in values if value)
    return counts.most_common(1)[0][0] if counts else None


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
    normalized_text, page_headers = Normalizer().normalize_with_metadata(loaded.raw_text)
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
    terminology_fn = terminology_extractor or (lambda s, d: TerminologyExtractor(min_mentions=2).extract(s, d))
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
        page_headers=page_headers,
    )
    logger.info("Preprocessing completed in %.3fs", perf_counter() - started)
    return result