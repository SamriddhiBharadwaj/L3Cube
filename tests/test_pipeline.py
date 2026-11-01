"""Focused tests for English preprocessing, extraction, and chunk boundaries."""

import asyncio
import re
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

    def test_normalizer_repairs_pdf_artifacts_and_extracts_page_metadata(self) -> None:
        text = (
            "Petitioner‟s father‟s birthday. Ms.Geeta met Sh.Sanjeev and gave a "
            "well- reasoned opinion. CM(M) 47/2022 & CM(M) 211/2022 Page 2 of 10 "
            "The order continues."
        )
        normalized, page_headers = Normalizer().normalize_with_metadata(text)

        self.assertIn("Petitioner's father's birthday", normalized)
        self.assertIn("Ms. Geeta", normalized)
        self.assertIn("Sh. Sanjeev", normalized)
        self.assertIn("well-reasoned", normalized)
        self.assertNotIn("CM(M)", normalized)
        self.assertEqual([header.page_number for header in page_headers], [2])

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

    def test_paragraph_splitter_recovers_line_start_enumerators(self) -> None:
        normalized = Normalizer().normalize(
            "1. First section text.\n2. Second section text.\n(i) Nested item.\n3. Third section."
        )
        paragraphs = ParagraphSplitter().split(normalized, "doc-enum")

        self.assertEqual(
            [paragraph.text.split()[0] for paragraph in paragraphs],
            ["1.", "2.", "(i)", "3."],
        )
        self.assertEqual(paragraphs[1].section_id, paragraphs[2].section_id)
        self.assertNotEqual(paragraphs[0].section_id, paragraphs[1].section_id)
        self.assertNotEqual(paragraphs[2].section_id, paragraphs[3].section_id)

    def test_sentence_splitter_uses_english_terminal_punctuation(self) -> None:
        paragraph = ParagraphSplitter().split(
            "Alice arrived. She asked! Then left? English is fine. Next sentence.", "doc-2"
        )
        paragraphs = SentenceSplitter().split(paragraph)

        self.assertEqual(len(paragraphs[0].sentences), 5)
        self.assertTrue(paragraphs[0].sentences[0].text.endswith("."))
        self.assertTrue(paragraphs[0].sentences[2].text.endswith("?"))

    def test_sentence_splitter_preserves_abbreviations_and_enumerator_ownership(self) -> None:
        paragraph = ParagraphSplitter().split(
            "The visit is at 10.00 a.m. in the morning. Mr. Smith and Ms. Jones "
            "reviewed No. 8 vs. No. 9. (i) First list item. 6. Sixth list item. "
            "11. Eleventh list item. (vii) Final item.",
            "doc-abbrev",
        )
        sentences = SentenceSplitter().split(paragraph)[0].sentences

        self.assertTrue(any("10.00 a.m. in the morning" in sentence.text for sentence in sentences))
        self.assertTrue(any("Mr. Smith" in sentence.text and "Ms. Jones" in sentence.text for sentence in sentences))
        self.assertEqual(
            [sentence.enumerator for sentence in sentences if sentence.enumerator],
            ["(i)", "6.", "11.", "(vii)"],
        )
        self.assertFalse(any(re.fullmatch(r"(?:\d+\.|\([ivxlcdm]+\))", sentence.text.strip()) for sentence in sentences))

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
                Entity("term", "doc", "London", None, "technical_term", source="yake"),
                Entity("ner", "doc", "London", None, "place"),
            ]
        )

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].termType, "place")

    def test_merge_deduplicates_surface_across_types_and_votes_by_mentions(self) -> None:
        candidates = [
            Entity("place-1", "doc", "Saket", None, "place", mentionCount=3, source="ner", confidence=0.91),
            Entity("person-1", "doc", "SAKET", None, "person", mentionCount=1, source="ner", confidence=0.83),
            Entity("term-1", "doc", "Saket", None, "technical_term", mentionCount=5, source="yake"),
            Entity("place-2", "doc", "South-East", None, "place", source="ner"),
            Entity("place-3", "doc", "South-East District", None, "place", source="ner"),
            Entity("person-2", "doc", "Kinri", None, "person", source="ner", confidence=0.92),
            Entity("org-1", "doc", "Kinri Dhir", None, "organization", source="ner", confidence=0.78),
        ]

        entities = merge_entity_candidates(candidates)
        by_term = {entity.sourceTerm: entity for entity in entities}

        self.assertEqual(by_term["Saket"].termType, "place")
        self.assertEqual(by_term["Saket"].mentionCount, 3)
        self.assertEqual(by_term["South-East District"].termType, "place")
        self.assertEqual(by_term["Kinri Dhir"].termType, "person")

    def test_entity_extractor_separates_identifiers_and_infers_honorific_gender(self) -> None:
        text = "Ms. Geeta Luthra discussed CM(M) 47/2022 and passport GA340868 on 22.12.2021."
        sentences = [Sentence("s1", text, "p1", "sec1", 1)]
        extractor = EntityExtractor()

        def fake_ner_pipeline(texts, **_kwargs):
            person_start = text.index("Geeta")
            person_end = text.index(" discussed")
            return [[
                {
                    "entity_group": "PER",
                    "word": "Geeta Luthra",
                    "start": person_start,
                    "end": person_end,
                    "score": 0.96,
                }
            ]]

        extractor._get_pipeline = lambda: fake_ner_pipeline
        entities = extractor.extract(sentences, "doc-evidence")
        by_type = {entity.termType: entity for entity in entities}

        self.assertEqual(by_type["person"].sourceTerm, "Geeta Luthra")
        self.assertEqual(by_type["person"].gender, "female")
        self.assertEqual(by_type["person"].confidence, 0.96)
        self.assertEqual(by_type["person"].firstSentenceId, "s1")
        self.assertEqual(by_type["case_number"].sourceTerm, "CM(M) 47/2022")
        self.assertEqual(by_type["identifier"].sourceTerm, "GA340868")
        self.assertEqual(by_type["date"].sourceTerm, "22.12.2021")

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
        normalized = Normalizer().normalize(SAMPLE)
        paragraphs = ParagraphSplitter().split(normalized, "doc-chunk")
        paragraphs = SentenceSplitter().split(paragraphs)
        return preprocess_document_from_parts(paragraphs, normalized)

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
        self.assertEqual("".join(chunk.srcText for chunk in chunks), self._preprocessed().normalized_text)
        self.assertTrue(all("⁇" not in chunk.srcText for chunk in chunks))

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

    def test_chunk_contexts_are_exact_source_sentence_substrings(self) -> None:
        source = " ".join(
            "Sentence{}.".format(index) for index in range(1, 41)
        )
        normalized = Normalizer().normalize(source)
        paragraphs = SentenceSplitter().split(ParagraphSplitter().split(normalized, "doc-context"))
        document = preprocess_document_from_parts(paragraphs, normalized)

        chunks = chunk_document(document, 16, tokenizer=WhitespaceTokenizer())

        self.assertGreater(len(chunks), 1)
        for index, chunk in enumerate(chunks):
            if index > 0:
                self.assertIn(chunk.precedingContext, chunks[index - 1].srcText)
            if index + 1 < len(chunks):
                self.assertIn(chunk.followingContext, chunks[index + 1].srcText)
            self.assertNotIn("⁇", chunk.precedingContext or "")
            self.assertNotIn("⁇", chunk.followingContext or "")
        self.assertEqual("".join(chunk.srcText for chunk in chunks), normalized)

    def test_golden_invariants_on_legal_document(self) -> None:
        """Verify the 5 golden invariants on real legal text with enumerators and citations."""
        legal_path = Path(__file__).resolve().parent.parent / "legal-3.txt"
        if not legal_path.exists():
            text = (
                "1. CM (M) 47/2022 has been filed. The petitioner Anaaya visited at 10.00 a.m. in the morning.\n\n"
                "2. While this petition was pending:\n\n"
                "(i) The applicant shall have visitation from 3.00 p.m. to 6.00 p.m. respectively.\n\n"
                "33. Quoted judgment passage.\n\n"
                "10. Final order."
            )
        else:
            text = TextLoader().load(str(legal_path)).raw_text

        normalized, headers = Normalizer().normalize_with_metadata(text)
        paragraphs = SentenceSplitter().split(ParagraphSplitter().split(normalized, "doc-golden"))
        document = preprocess_document_from_parts(paragraphs, normalized)

        chunks = chunk_document(document, 256, tokenizer=WhitespaceTokenizer())

        self.assertGreater(len(chunks), 1)

        bare_enum_regex = re.compile(
            r"(?:^|[\s:;])(?:\d{1,2}\.|\([a-z0-9ivx]+\)|(?<![a|p]\.)\b[ivx]{1,4}\.|(?<![a|p]\.)\b[a-z]\.)\s*$",
            re.IGNORECASE,
        )

        all_sentences = [sentence for paragraph in document.paragraphs for sentence in paragraph.sentences]
        assigned_sentence_ids = set()

        # Invariant 3: Concatenating all srcText in order reproduces normalized text
        reconstructed = "".join(chunk.srcText for chunk in chunks)
        self.assertEqual(reconstructed, normalized)

        for index, chunk in enumerate(chunks):
            # Invariant 1: precedingContext is exact substring of previous chunk's srcText, following of next
            if chunk.precedingContext is not None:
                self.assertGreater(index, 0)
                self.assertIn(chunk.precedingContext, chunks[index - 1].srcText)
            if chunk.followingContext is not None:
                self.assertLess(index, len(chunks) - 1)
                self.assertIn(chunk.followingContext, chunks[index + 1].srcText)

            # Invariant 2: No chunk ends with a bare enumerator
            self.assertIsNone(
                bare_enum_regex.search(chunk.srcText.strip()),
                f"Chunk {index} ends with a bare enumerator: {chunk.srcText[-30:]!r}",
            )

            # Invariant 5: No ⁇ in srcText, precedingContext, or followingContext
            self.assertNotIn("⁇", chunk.srcText)
            self.assertNotIn("⁇", chunk.precedingContext or "")
            self.assertNotIn("⁇", chunk.followingContext or "")

            # Invariant 4: Every sentence belongs to exactly one chunk
            self.assertTrue(len(chunk.sentences) > 0, f"Chunk {index} has no sentences")
            for sentence in chunk.sentences:
                self.assertEqual(sentence.chunk_id, chunk.chunkId)
                self.assertNotIn(sentence.sentence_id, assigned_sentence_ids)
                assigned_sentence_ids.add(sentence.sentence_id)

        self.assertEqual(len(assigned_sentence_ids), len(all_sentences))

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


def preprocess_document_from_parts(paragraphs, normalized_text=None) -> PreprocessedDocument:
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
        normalized_text if normalized_text is not None else Normalizer().normalize(SAMPLE),
        paragraphs,
        LanguageDetection("en", 1.0, True),
        [],
        len(normalized_text if normalized_text is not None else SAMPLE),
        (normalized_text if normalized_text is not None else SAMPLE).count("\n") + 1,
    )


if __name__ == "__main__":
    unittest.main()