# L3Cube Translation Pipeline Overview

## 1. What this repo does

This project is a document preprocessing and chunking pipeline for legal English source text, designed to prepare source content for downstream translation into Indic languages such as Hindi.

At a high level, the flow is:

1. Load a UTF-8 `.txt` source document.
2. Normalize and split it into paragraphs and sentences.
3. Detect whether the text is English and flag mixed-script or code-switch segments.
4. Extract entity and terminology candidates from the English text.
5. Chunk the document into translation-sized units with context preserved separately from the source text.
6. Write a structured report containing chunk metadata and the entity/term registry.

This is a preprocessing pipeline first; it is not yet a full end-to-end multilingual translation system.

---

## 2. Repository structure

- `tester.py` — CLI entrypoint for running the pipeline on a source document.
- `pipeline_output.txt` — generated output report from a run.
- `legal-3.txt` — current English legal source sample.
- `legal-3__hindi.txt` — older Hindi-based sample kept as reference/legacy input.
- `translation_pipeline/` — core application code.
- `tests/` — project tests covering preprocessing, entity extraction, and chunking logic.
- `pyproject.toml` and `requirements.txt` — environment and dependency metadata.

---

## 3. Core modules and responsibilities

### `tester.py`

This is the user-facing runner.

It accepts a required input file and an optional target language override:

```bash
.venv312/bin/python tester.py legal-3.txt
.venv312/bin/python tester.py legal-3.txt --target-language hin_Deva
```

The script calls `run_preprocessing_pipeline(...)`, collects ordered chunks, and writes a summary report to `pipeline_output.txt`.

The output report includes:

- document ID
- chunk count
- entity/term registry count
- chunk metadata such as section, paragraph, context, and source text
- entity candidates and their mapped types

### `translation_pipeline/__init__.py`

This file exposes the public package API for the pipeline. It re-exports:

- `preprocess_document`
- `chunk_document`
- `run_preprocessing_pipeline`
- dataclasses such as `DocumentRecord`, `Entity`, `TranslationChunk`, and `PreprocessedDocument`

This makes the package look like a stable import surface for the rest of the application.

### `translation_pipeline/models.py`

This file defines the shared data model used across all pipeline stages.

Key records:

- `LoadedText` — raw document text and basic size metadata
- `Sentence` — sentence text with paragraph/section IDs
- `Paragraph` — paragraph metadata and sentence list
- `DocumentRecord` — eventual database row for document metadata
- `LanguageDetection` — detected language and code-switch flags
- `Entity` — extracted entity or term candidate
- `PreprocessedDocument` — full document after normalization and extraction
- `TranslationChunk` — chunk record carrying source text and surrounding context

This is the central type contract for the whole pipeline.

### `translation_pipeline/preprocessing.py`

This is the main document-cleaning and extraction stage.

It contains:

- `TextLoader` — reads and validates UTF-8 `.txt` inputs
- `Normalizer` — strips control characters, normalizes punctuation, collapses repeated blank lines
- `ParagraphSplitter` — groups source text into paragraphs and section IDs
- `SentenceSplitter` — splits on English sentence punctuation
- `LanguageDetector` — verifies English source text and flags mixed-script/text issues
- `EntityExtractor` — runs named-entity extraction over English sentences
- terminology extraction helpers — collect important phrases/terms
- candidate merging logic — collapse duplicate or alias variants

The file is designed to enforce an English-source workflow while still tracking mixed-language or low-confidence segments.

### `translation_pipeline/chunking.py`

This is the chunking / token-budget stage.

It includes:

- `IndicTrans2Tokenizer` — wraps the IndicTrans2 tokenizer used for source-token counting and language-tagged encoding
- `TextTokenizer` protocol — minimal interface expected by the chunker
- `TokenBudgetManager` — reserves context space while enforcing target max-token budgets
- `SemanticChunker` — groups full sentences into chunks and respects paragraph boundaries
- `OverlapInjector` — adds preceding/following context without changing the actual source text
- `ChunkBuilder` — creates the final `TranslationChunk` objects

This stage is important because it keeps context outside the actual `srcText`, which is a useful pattern for translation systems.

### `translation_pipeline/pipeline.py`

This module orchestrates the whole preprocess → chunking handoff.

It exposes:

- `TranslationChunkQueue` — async queue for ordered chunk delivery
- `run_preprocessing_pipeline(...)` — runs preprocessing and chunking, then returns a queue carrying the document’s chunks and entity registry

This is effectively the “main pipeline coordinator” for the project.

---

## 4. Data flow

The normal runtime flow looks like this:

```text
source .txt file
  -> TextLoader
  -> Normalizer
  -> ParagraphSplitter
  -> SentenceSplitter
  -> LanguageDetector
  -> EntityExtractor + terminology extraction
  -> merge entity candidates
  -> PreprocessedDocument
  -> chunk_document(...)
  -> TranslationChunkQueue
  -> tester.py writes pipeline_output.txt
```

The main contract is:

- `PreprocessedDocument` carries the cleaned, annotated document state
- `TranslationChunk` carries translation units and their neighboring context
- `Entity` captures the document-level registry of extracted terms

---

## 5. Current design direction

The repository was shifted from a Hindi-first assumption to an English-source preprocessing pipeline. In practical terms:

- source validation now expects English content
- sentence and paragraph logic are English-oriented
- NER extraction is built around English entity labels
- the chunker still targets Indic language codes such as `hin_Deva` via IndicTrans2 tokenizer metadata

This means the system is structured as a legal document processing batch for English input, with a downstream Indic translation target in mind.

---

## 6. How to run it

Use the project virtual environment and call the tester script with the document path:

```bash
cd "/Users/samriddhibharadwaj/Documents/School/L3CubeTranslationPipeline"
.venv312/bin/python tester.py legal-3.txt
```

Optional target language override:

```bash
.venv312/bin/python tester.py legal-3.txt --target-language hin_Deva
```

The script writes the report to:

```text
pipeline_output.txt
```

---

## 7. Verified baseline status

I verified the CLI and test baseline directly in the repo.

Command used:

```bash
.venv312/bin/python tester.py --help
.venv312/bin/python -m unittest discover -s tests -v
```

Observed result:

- CLI help prints the correct usage with the positional `input_file` and optional `--target-language`
- the test suite completed with `Ran 19 tests ... OK`

---

## 8. Important known limitation

The pipeline logic is in a healthy, tested state, but the actual model-backed parts are still gated by external Hugging Face access and checkpoint availability.

A key issue in this project is that the tokenizer and related model resources may require:

- Hugging Face access to gated checkpoints
- authentication via `hf auth login`
- direct model downloads from the Hugging Face Hub

The repo’s chunking code points to:

```text
ai4bharat/indictrans2-en-indic-1B
```

This is the main external dependency that can block end-to-end execution if access is not granted or if the download is interrupted.

---

## 9. Short interpretation

This repository is best understood as:

- a preprocessing and entity-discovery layer for English legal documents
- a chunking engine that respects token budgets and context windows
- a project scaffold for feeding extracted and segmented legal text into a translation system

It is not simply a translation script; it is a document preparation pipeline with an explicit “translation unit” abstraction and a structured report output.

---

## 10. Practical next steps

If the goal is to continue building this project, the most important next steps are:

1. resolve model access for the IndicTrans2 tokenizer/checkpoint
2. stabilize NER and terminology extraction quality on English legal text
3. decide whether NER-only extraction is sufficient or whether an LLM should be used as a constrained validation layer
4. feed the resulting chunks into an actual Indic translation step

The architecture is already set up to support that path, but the external model checkpoint access remains the main blocker to full execution.
