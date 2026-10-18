"""Focused tests for English preprocessing, extraction, and chunk boundaries."""

import asyncio
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from typing import List, Sequence
from unittest.mock import patch

from translation_pipeline.chunking import (
    ChunkDraft,
    ChunkWithContext,
    IndicTrans2Tokenizer,
    OverlapInjector,
    TextTokenizer,
    TokenBudgetManager,
    chunk_document,
)
from translation_pipeline.models import (
    Entity,
    LanguageDetection,
    PreprocessedDocument,
    Sentence,
)
from translation_pipeline.pipeline import run_preprocessing_pipeline
from translation_pipeline.preprocessing import (
    EntityExtractor,
    LanguageDetector,
    Normalizer,
    ParagraphSplitter,
    SentenceSplitter,
    TextLoader,
    TerminologyExtractor,
    merge_entity_candidates,
    preprocess_document,
)


SAMPLE = (
    "Chapter 1\n\n"
    "Alice Johnson visited London. She reviewed the AI model and spoke with Acme Labs. "
    "It works! Then she returned home."
)


class WhitespaceTokenizer:
    """Deterministic test tokenizer; production always uses IndicTrans2."""

    def encode(self, text: str) -> List[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join("token{}".format(token_id) for token_id in token_ids)


class EnglishPreprocessingTests(unittest.TestCase):
    def test_text_loader_reads_utf8_and_counts_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "sample.txt"
            file_path.write_text("Hello\nworld", encoding="utf-8")
            loaded = TextLoader().load(str(file_path))

        self.assertEqual(loaded.raw_text, "Hello\nworld")
        self.assertEqual(loaded.char_count, len("Hello\nworld"))
        self.assertEqual(loaded.line_count, 2)

    def test_normalizer_preserves_unicode_and_cleans_punctuation(self) -> None:
        text = "Cafe\u0301  “AI”—model…\x01\r\n"
        normalized = Normalizer().normalize(text)

        self.assertIn("Café", normalized)
        self.assertIn('"AI"-model...', normalized)
        self.assertNotIn("\x01", normalized)

    def test_paragraph_splitter_assigns_english_heading_sections(self) -> None:
        normalized = Normalizer().normalize(
            "Chapter 1\n\nFirst paragraph.\n  Indented paragraph.\n\n"
            "Chapter 2\n\nFinal paragraph."
        )
        paragraphs = ParagraphSplitter().split(normalized, "doc-1")

        self.assertEqual(len(paragraphs), 5)
        self.assertTrue(paragraphs[0].is_heading)
        self.assertEqual(paragraphs[0].section_id, paragraphs[1].section_id)
        self.assertEqual(paragraphs[1].section_id, paragraphs[2].section_id)
        self.assertNotEqual(paragraphs[2].section_id, paragraphs[3].section_id)
        self.assertNotEqual(paragraphs[1].paragraph_id, paragraphs[2].paragraph_id)

    def test_sentence_splitter_uses_english_terminal_punctuation(self) -> None:
        paragraph = ParagraphSplitter().split(
            "Alice arrived. She asked! Then left? English is fine. Next sentence.", "doc-2"
        )
        paragraphs = SentenceSplitter().split(paragraph)

        self.assertEqual(len(paragraphs[0].sentences), 5)
        self.assertTrue(paragraphs[0].sentences[0].text.endswith("."))
        self.assertTrue(paragraphs[0].sentences[2].text.endswith("?"))

    def test_language_detector_confirms_english_and_flags_mixed_script(self) -> None:
        paragraphs = ParagraphSplitter().split("Alice visited London. नमस्ते, Alice.", "doc-3")
        sentences = [sentence for paragraph in SentenceSplitter().split(paragraphs) for sentence in paragraph.sentences]

        detection = LanguageDetector().detect(sentences)

        self.assertTrue(detection.is_english)
        self.assertTrue(any("नमस्ते" in flag.text for flag in detection.code_switch_flags))

    def test_preprocessing_extractors_run_concurrently_and_merge_entities(self) -> None:
        barrier = threading.Barrier(3)
        expected_language = LanguageDetection("en", 0.99, True, [])

        def detect(_sentences):
            barrier.wait(timeout=2)
            return expected_language

        def extract_ner(_sentences, document_id):
            barrier.wait(timeout=2)
            return [Entity("ner-1", document_id, "Alice Johnson", None, "person")]

        def extract_terms(_sentences, document_id):
            barrier.wait(timeout=2)
            return [
                Entity("term-1", document_id, "Alice Johnson", None, "technical_term"),
                Entity("term-2", document_id, "AI model", None, "technical_term"),
            ]

        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "narrative.txt"
            file_path.write_text(SAMPLE, encoding="utf-8")
            result = preprocess_document(
                str(file_path),
                "doc-4",
                target_language="hin_Deva",
                language_detector=detect,
                entity_extractor=extract_ner,
                terminology_extractor=extract_terms,
            )

        self.assertEqual(result.document.sourceLanguage, "eng_Latn")
        self.assertEqual(result.document.targetLanguage, "hin_Deva")
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(result.entities[0].termType, "person")
        self.assertIsNone(result.entities[0].targetName)
        self.assertGreater(len(result.paragraphs[1].sentences), 1)

    def test_merge_entity_candidates_prefers_named_entity_type(self) -> None:
        entities = merge_entity_candidates(
            [
                Entity("term", "doc", "London", None, "technical_term"),
                Entity("ner", "doc", "London", None, "place"),
            ]
        )

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].termType, "place")

    def test_merge_entity_candidates_collapses_english_spelling_aliases(self) -> None:
        candidates = [
            Entity("person-1", "doc", "Aditi", None, "person"),
            Entity("person-2", "doc", "Aditii", None, "person"),
            Entity("person-3", "doc", "Abhishek Ahuja", None, "person"),
            Entity("person-4", "doc", "Abhishek Ahujaa", None, "person"),
            Entity("person-5", "doc", "Bakht", None, "person"),
            Entity("person-6", "doc", "Amity", None, "person"),
        ]

        entities = merge_entity_candidates(candidates)

        self.assertEqual(len(entities), 4)
        self.assertEqual(entities[0].sourceTerm, "Aditii")
        self.assertEqual(entities[1].sourceTerm, "Abhishek Ahujaa")
        self.assertEqual(entities[2].sourceTerm, "Bakht")
        self.assertEqual(entities[3].sourceTerm, "Amity")

    def test_merge_entity_candidates_uses_unique_full_name_for_person_aliases(self) -> None:
        candidates = [
            Entity("full", "doc", "Anaya Ahuja", None, "person"),
            Entity("short", "doc", "Anaya", None, "person"),
            Entity("variant", "doc", "Anayaa", None, "person"),
            Entity("other", "doc", "Amity", None, "person"),
        ]

        entities = merge_entity_candidates(candidates)

        self.assertEqual([entity.sourceTerm for entity in entities], ["Anaya Ahuja", "Amity"])

    def test_entity_extractor_maps_english_ner_tags(self) -> None:
        paragraphs = ParagraphSplitter().split(SAMPLE, "doc-ner")
        sentences = [sentence for paragraph in SentenceSplitter().split(paragraphs) for sentence in paragraph.sentences]
        extractor = EntityExtractor()
        observed = {}

        def fake_ner_pipeline(texts, **kwargs):
            observed["texts"] = texts
            observed["kwargs"] = kwargs
            return [[
                {"entity_group": "PER", "word": "Alice Johnson"},
                {"entity_group": "LOC", "word": "London"},
                {"entity_group": "ORG", "word": "Acme Labs"},
                {"entity_group": "MISC", "word": "AI"},
            ]]

        extractor._get_pipeline = lambda: fake_ner_pipeline

        entities = extractor.extract(sentences, "doc-ner")

        self.assertEqual(
            [entity.termType for entity in entities],
            ["person", "place", "organization", "proper_noun"],
        )
        self.assertTrue(all(entity.targetName is None for entity in entities))
        self.assertEqual(observed["texts"], [sentence.text for sentence in sentences])
        self.assertEqual(observed["kwargs"], {"batch_size": 4})

    def test_entity_extractor_rebuilds_wordpieces_from_source_offsets(self) -> None:
        text = "Abhishek Ahuja met Aditi Shah"
        first_name_end = text.index(" ")
        surname_start = text.index("Ahuja")
        surname_end = surname_start + len("Ahuja")
        second_name_start = text.index("Aditi")
        second_name_end = text.index(" Shah") + len(" Shah")
        predictions = [
            {"entity_group": "PER", "word": "Ab", "start": 0, "end": 2, "score": 0.95},
            {"entity_group": "PER", "word": "##hishek", "start": 2, "end": first_name_end, "score": 0.90},
            {"entity_group": "PER", "word": "Ah", "start": surname_start, "end": surname_start + 2, "score": 0.90},
            {"entity_group": "PER", "word": "##uja", "start": surname_start + 2, "end": surname_end, "score": 0.90},
            {"entity_group": "PER", "word": "Aditi", "start": second_name_start, "end": second_name_end, "score": 0.90},
        ]
        extractor = EntityExtractor()
        extractor._get_pipeline = lambda: lambda _texts, **_kwargs: [predictions]

        entities = extractor.extract(
            [Sentence("s1", text, "p1", "sec1", 1)],
            "doc-wordpieces",
        )

        self.assertEqual([entity.sourceTerm for entity in entities], ["Abhishek Ahuja", "Aditi Shah"])

    def test_entity_extractor_rejects_english_role_words_as_people(self) -> None:
        text = "The petitioner appeared before the court."
        predictions = [
            {"entity_group": "PER", "word": term, "start": text.index(term),
             "end": text.index(term) + len(term), "score": 0.95}
            for term in ("petitioner",)
        ]
        extractor = EntityExtractor()
        extractor._get_pipeline = lambda: lambda _texts, **_kwargs: [predictions]

        entities = extractor.extract([Sentence("s1", text, "p1", "sec1", 1)], "doc-noise")

        self.assertEqual(entities, [])

    def test_terminology_extractor_removes_stopwords_and_short_noise(self) -> None:
        paragraphs = ParagraphSplitter().split(SAMPLE, "doc-terms")
        sentences = [sentence for paragraph in SentenceSplitter().split(paragraphs) for sentence in paragraph.sentences]

        class FakeKeywordExtractor:
            def __init__(self, **_kwargs):
                pass

            def extract_keywords(self, _text):
                return [
                    ("the", 0.01),
                    ("of the", 0.02),
                    ("family court", 0.03),
                    ("the family court", 0.04),
                    ("AI model", 0.05),
                    ("in the court", 0.06),
                ]

        with patch.dict(sys.modules, {"yake": types.SimpleNamespace(KeywordExtractor=FakeKeywordExtractor)}):
            entities = TerminologyExtractor().extract(sentences, "doc-terms")

        self.assertEqual([entity.sourceTerm for entity in entities], [
            "family court",
            "family court",
            "AI model",
        ])

    def test_terminology_extractor_maps_yake_keywords(self) -> None:
        paragraphs = ParagraphSplitter().split(SAMPLE, "doc-yake")
        sentences = [sentence for paragraph in SentenceSplitter().split(paragraphs) for sentence in paragraph.sentences]

        class FakeKeywordExtractor:
            def __init__(self, **_kwargs):
                pass

            def extract_keywords(self, _text):
                return [("AI model", 0.01)]

        with patch.dict(sys.modules, {"yake": types.SimpleNamespace(KeywordExtractor=FakeKeywordExtractor)}):
            entities = TerminologyExtractor().extract(sentences, "doc-yake")

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].sourceTerm, "AI model")
        self.assertEqual(entities[0].termType, "technical_term")


class ChunkingTests(unittest.TestCase):
    def _preprocessed(self) -> PreprocessedDocument:
        paragraphs = ParagraphSplitter().split(SAMPLE, "doc-chunk")
        paragraphs = SentenceSplitter().split(paragraphs)
        return preprocess_document_from_parts(paragraphs)

    def test_budget_reserves_space_for_context(self) -> None:
        budget = TokenBudgetManager(100)

        self.assertEqual(budget.reserved_tokens, 12)
        self.assertEqual(budget.content_budget, 88)
        self.assertEqual(budget.overlap_tokens, 6)

    def test_indictrans_adapter_adds_language_tags_and_decodes_source_tokens(self) -> None:
        class FakeSentencePiece:
            def DecodePieces(self, pieces):
                return " ".join(pieces)

        class FakeIndicTokenizer:
            src_decoder = {12: "Hello", 13: "world"}
            unk_token = "<unk>"
            src_spm = FakeSentencePiece()

            def __init__(self):
                self.input_text = None
                self.verbose = None

            def encode(self, text, add_special_tokens, verbose):
                self.input_text = text
                self.verbose = verbose
                return [10, 11, 12, 13]

        fake_tokenizer = FakeIndicTokenizer()
        adapter = IndicTrans2Tokenizer(fake_tokenizer, target_language="hin_Deva")

        token_ids = adapter.encode("Hello world")

        self.assertEqual(fake_tokenizer.input_text, "eng_Latn hin_Deva Hello world")
        self.assertFalse(fake_tokenizer.verbose)
        self.assertEqual(token_ids, [10, 11, 12, 13])
        self.assertEqual(adapter.decode(token_ids[2:]), "Hello world")

    def test_chunking_keeps_context_separate_and_respects_limit(self) -> None:
        tokenizer = WhitespaceTokenizer()
        chunks = chunk_document(self._preprocessed(), 16, tokenizer=tokenizer)

        self.assertGreater(len(chunks), 1)
        self.assertIsNone(chunks[0].precedingContext)
        self.assertIsNone(chunks[-1].followingContext)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.unitIndex, index)
            self.assertEqual(chunk.status, "pending")
            self.assertEqual(chunk.repairCount, 0)
            self.assertLessEqual(
                len(
                    tokenizer.encode(
                        " ".join(
                            text
                            for text in (
                                chunk.precedingContext,
                                chunk.srcText,
                                chunk.followingContext,
                            )
                            if text
                        )
                    )
                ),
                16,
            )
        self.assertTrue(any(chunk.followingContext for chunk in chunks[:-1]))

    def test_context_injector_only_sets_neighbor_fields(self) -> None:
        tokenizer = WhitespaceTokenizer()
        drafts = [
            ChunkDraft("section", "p1", "one two three"),
            ChunkDraft("section", "p2", "four five six"),
        ]
        chunks = OverlapInjector().inject(drafts, tokenizer, 1, 20)

        self.assertEqual(chunks[0].draft.src_text, "one two three")
        self.assertIsNone(chunks[0].preceding_context)
        self.assertIsNotNone(chunks[0].following_context)
        self.assertIsNotNone(chunks[1].preceding_context)
        self.assertIsNone(chunks[1].following_context)

    def test_async_handoff_populates_ordered_queue_and_registry(self) -> None:
        barrier = threading.Barrier(3)

        def detect(_sentences):
            barrier.wait(timeout=2)
            return LanguageDetection("en", 1.0, True, [])

        def extract_entity(_sentences, document_id):
            barrier.wait(timeout=2)
            return [Entity("entity", document_id, "Alice Johnson", None, "person")]

        def extract_terms(_sentences, _document_id):
            barrier.wait(timeout=2)
            return []

        async def run():
            with tempfile.TemporaryDirectory() as directory:
                file_path = Path(directory) / "narrative.txt"
                file_path.write_text(SAMPLE, encoding="utf-8")
                return await run_preprocessing_pipeline(
                    str(file_path),
                    "doc-queue",
                    16,
                    tokenizer=WhitespaceTokenizer(),
                    language_detector=detect,
                    entity_extractor=extract_entity,
                    terminology_extractor=extract_terms,
                )

        queue = asyncio.run(run())
        self.assertEqual(len(queue.entities), 1)
        chunks = []
        while not queue.empty():
            chunk = queue.get_nowait()
            chunks.append(chunk)
            queue.task_done()
        self.assertEqual([chunk.unitIndex for chunk in chunks], list(range(len(chunks))))


def preprocess_document_from_parts(paragraphs) -> PreprocessedDocument:
    """Make a preprocessed fixture without invoking external extraction models."""
    from datetime import datetime, timezone

    from translation_pipeline.models import DocumentRecord

    now = datetime.now(timezone.utc)
    document = DocumentRecord(
        "doc-chunk",
        "sample",
        "eng_Latn",
        "hin_Deva",
        None,
        None,
        "preprocessed",
        now,
        now,
    )
    return PreprocessedDocument(
        document,
        SAMPLE,
        paragraphs,
        LanguageDetection("en", 1.0, True),
        [],
        len(SAMPLE),
        1,
    )


if __name__ == "__main__":
    unittest.main()