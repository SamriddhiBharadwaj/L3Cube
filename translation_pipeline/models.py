"""Typed intermediate records shared by preprocessing and chunking stages."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class LoadedText:
    """Raw UTF-8 document contents and inexpensive file metadata."""

    raw_text: str
    char_count: int
    line_count: int


@dataclass
class Sentence:
    """An English sentence and its position within a document paragraph."""

    sentence_id: str
    text: str
    paragraph_id: str
    section_id: str
    sentence_index: int
    char_start: int = 0
    char_end: int = 0
    enumerator: Optional[str] = None
    chunk_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class Paragraph:
    """A database-friendly paragraph record with its constituent sentences."""

    paragraph_id: str
    section_id: str
    text: str
    paragraph_index: int
    is_heading: bool = False
    sentences: List[Sentence] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    enumerator: Optional[str] = None


@dataclass
class PageHeader:
    """Page-label metadata removed from PDF-derived source text."""

    page_number: int
    total_pages: int
    header_text: str


@dataclass
class DocumentRecord:
    """Document metadata matching the eventual relational Document row."""

    documentId: str
    title: str
    sourceLanguage: str
    targetLanguage: str
    genre: Optional[str]
    register: Optional[str]
    status: str
    createdAt: datetime
    updatedAt: datetime


@dataclass
class CodeSwitchFlag:
    """A sentence or segment requiring mixed-language downstream handling."""

    sentence_id: str
    text: str
    detected_language: str
    confidence: float
    reason: str


@dataclass
class LanguageDetection:
    """Document-level English language result plus code-switch flags."""

    detected_language: str
    confidence: float
    is_english: bool
    code_switch_flags: List[CodeSwitchFlag] = field(default_factory=list)


@dataclass
class Entity:
    """Entity/term candidate matching the eventual relational Entity row."""

    entityId: str
    documentId: str
    sourceTerm: str
    targetName: Optional[str]
    termType: str
    gender: Optional[str] = None
    mentionCount: int = 1
    firstSentenceId: Optional[str] = None
    source: str = "unknown"
    confidence: Optional[float] = None


@dataclass
class PreprocessedDocument:
    """Complete result of one-time document preprocessing."""

    document: DocumentRecord
    normalized_text: str
    paragraphs: List[Paragraph]
    language: LanguageDetection
    entities: List[Entity]
    char_count: int
    line_count: int
    page_headers: List[PageHeader] = field(default_factory=list)


@dataclass
class TranslationChunk:
    """A translation unit matching the eventual TranslationChunk database row."""

    chunkId: str
    documentId: str
    sectionId: str
    paragraphId: str
    unitIndex: int
    srcText: str
    precedingContext: Optional[str]
    followingContext: Optional[str]
    status: str = "pending"
    repairCount: int = 0
    contentTokenCount: int = 0
    contextTokenCount: int = 0
    unknownTokenRate: float = 0.0
    warnings: List[str] = field(default_factory=list)
    sentences: List[Sentence] = field(default_factory=list)