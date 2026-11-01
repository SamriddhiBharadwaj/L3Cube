# Claude Handoff: L3Cube Translation Pipeline

## Purpose

This repository is the first-stage preparation layer for a long-document translation project. Its current direction is **English legal-document input** with a configurable Indic target language. It loads a UTF-8 `.txt` file, normalizes and segments it, detects English, extracts entity/term candidates, chunks complete sentences under a model-token budget, and writes a readable report. Translation generation, consistency repair, database persistence, and final document assembly are not implemented.

The project originally began as Hindi-source preprocessing. The user later corrected the source-language requirement to English. `legal-3.txt` is the English sample; `legal-3__hindi.txt` is legacy input and should not be passed to the current English pipeline.

## Repository Map

- `tester.py` — CLI runner; takes an English input file and optional target language; writes `pipeline_output.txt`.
- `translation_pipeline/models.py` — shared dataclasses for document, paragraph, sentence, language result, entity, preprocessed document, and chunk records.
- `translation_pipeline/preprocessing.py` — UTF-8 loading, Unicode/whitespace normalization, English paragraph and sentence splitting, language detection, English NER, YAKE terminology extraction, and candidate normalization/merging.
- `translation_pipeline/chunking.py` — IndicTrans2 tokenization, context budget reservation, sentence-safe chunk construction, overlap fields, and relational-shape chunk records.
- `translation_pipeline/pipeline.py` — async orchestration and an ordered `asyncio.Queue`; merged entities are exposed as `queue.entities`.
- `translation_pipeline/__init__.py` — public API exports.
- `tests/test_pipeline.py` — deterministic tests using injected extractor outputs and a fake tokenizer.
- `requirements.txt` — runtime libraries and Transformers compatibility range.
- `pyproject.toml` — package metadata and Python >=3.11 requirement.
- `REPO_OVERVIEW.md` — an earlier architecture overview; it may not reflect all recent model changes. Treat this handoff as the more current implementation-status summary.

## Runtime Flow

1. `tester.py` parses `input_file` and `--target-language` (default `hin_Deva`).
2. `run_preprocessing_pipeline(...)` calls synchronous preprocessing and chunking in worker threads with `asyncio.to_thread`.
3. `TextLoader` requires a `.txt` suffix and strict UTF-8 decoding.
4. `Normalizer` applies NFC, line-ending and whitespace cleanup, punctuation normalization, and control-character removal.
5. `ParagraphSplitter` creates paragraph and section IDs, recognizing simple English headings such as Chapter, Section, Part, Appendix, Introduction, Conclusion, and Summary.
6. `SentenceSplitter` splits at English `.`, `!`, and `?` followed by whitespace.
7. Three independent stages run concurrently using `ThreadPoolExecutor`: English language detection, NER, and YAKE terminology extraction.
8. Candidate entities are reconstructed from NER offsets, filtered, normalized/deduplicated, and combined with terminology candidates.
9. `IndicTrans2Tokenizer` prefixes `eng_Latn` and the chosen target code for token counting. `TokenBudgetManager` reserves 12% of `max_tokens` for overlap. The chunker groups full sentences, prefers paragraph boundaries, and raises if one sentence itself exceeds the content budget.
10. `OverlapInjector` stores preceding and following context in separate fields; it does not append context to `srcText`.
11. `ChunkBuilder` assigns UUIDs, sequential zero-based `unitIndex`, `status="pending"`, and `repairCount=0`.
12. `TranslationChunkQueue` is pre-populated in `unitIndex` order and carries `queue.entities` alongside the chunks.
13. `tester.py` writes labeled chunk and entity sections to `pipeline_output.txt`.

## Data Contracts

`DocumentRecord` has `documentId`, title, `sourceLanguage="eng_Latn"`, target language, genre, register, status, and timestamps.

`Entity` has `entityId`, `documentId`, `sourceTerm`, `targetName`, `termType`, and `gender`. Types currently include `person`, `place`, `organization`, `proper_noun`, and `technical_term`. `targetName` and `gender` are not inferred by the current extraction models, so usually remain `None`.

`TranslationChunk` has `chunkId`, document/section/paragraph IDs, `unitIndex`, `srcText`, separate `precedingContext` and `followingContext`, `status`, and `repairCount`.

## Current Model Choices and Constraints

- **English NER:** current code uses `elastic/distilbert-base-cased-finetuned-conll03-english`, a public 65M-parameter English CoNLL-2003 NER checkpoint. Metadata for its `model.safetensors` was verified as 260,803,668 bytes. It is much smaller than the earlier `dslim/bert-base-NER` checkpoint that displayed a 433 MB download. However, successful download and real inference with the smaller replacement have **not** been verified; a live load attempt was cancelled while stuck at 0%.
- **NER behavior:** maps PER/LOC/ORG/MISC to person/place/organization/proper_noun; uses batches of four; reconstructs subword predictions from offsets; filters a few known legal-role false positives. This remains a general-news NER model, not legal-domain NER.
- **Terminology:** YAKE runs with `lan="en"`; a small English stopword list removes function-word-only candidates and candidates with fewer than two remaining words.
- **Entity aliases:** case/accent/punctuation-normalized keys plus conservative same-token-count fuzzy matching merge near-identical English spellings. A one-word person alias can join a full name only when there is exactly one matching full-name candidate. This is heuristic and needs evaluation on more documents.
- **IndicTrans2 tokenizer:** current code points to `ai4bharat/indictrans2-en-indic-1B` and encodes source as `eng_Latn` plus target language. Earlier requests returned 403; the user later said access is available, but a successful English-checkpoint tokenizer load has not yet been verified. The tokenizer is all chunking needs; the pipeline does not load IndicTrans2 translation weights or translate text.
- **Transformers:** `requirements.txt` pins `transformers>=4.51,<5`; Transformers 5.x broke the remote IndicTrans2 config by removing `transformers.onnx`.
- **Hardware:** NER selects MPS when `torch.backends.mps.is_available()` and falls back to CPU when initialization/inference fails. No CUDA is assumed.
- **Privacy:** the source examples are legal documents. Any proposed hosted LLM review must be checked against data/privacy constraints; a local model or an explicit opt-in may be preferable.

## LLM Entity Review: Discussed, Not Implemented

The user asked whether noisy entity results should use an LLM instead of NER. Recommendation was a **hybrid** approach: NER supplies candidate labels/spans, then an LLM reviews candidates with document/sentence context, resolves aliases, and validates entity type. Every LLM result should be constrained to evidence spans from the source and validated against the input to reduce hallucinations. YAKE remains a separate term-candidate generator.

The user approved proceeding with that approach, but the follow-up request switched to documenting the repository. **No LLM client, prompt, reviewer stage, configuration, or tests have been added yet.** Treat this as an outstanding next feature, not an existing capability.

## Runner and Report

CLI:

```bash
.venv312/bin/python tester.py legal-3.txt
.venv312/bin/python tester.py legal-3.txt --target-language hin_Deva
```

The report is always written next to `tester.py` as `pipeline_output.txt`. It labels each chunk and its separate context fields, then lists entity/term candidate fields. Rerunning the script replaces that generated report. The checked-in/current report may be stale; do not treat it as a fresh English run unless the tester completed successfully after the English pivot.

## Verification Snapshot

Most recent verified test baseline from the workspace:

```bash
.venv312/bin/python tester.py --help
.venv312/bin/python -m unittest discover -s tests -v
```

Observed: CLI help showed the positional `input_file` and optional target flag; **19 unit tests passed**. `git status --short --ignored` showed `.DS_Store`, both virtual environments, `pipeline_output.txt`, and Python `__pycache__` directories ignored. `origin` is configured as `https://github.com/SamriddhiBharadwaj/L3Cube.git`.

Important test limitation: the NER, YAKE, and tokenizer tests primarily inject fakes. The last fresh unit run verifies deterministic processing logic, not live model downloads/inference. The latest smaller-NER download was reported stuck at 0%, and no fresh end-to-end English run has been verified since the source-language pivot.

## Recommended Next Steps

1. Resolve the stuck NER weight transfer. Start with `elastic/distilbert-base-cased-finetuned-conll03-english` (about 261 MB according to Hub metadata), but do not claim it works until a fresh download and sample inference succeed. The earlier 433 MB attempt was the previous BERT-base checkpoint.
2. Verify `hf auth whoami` uses the approved account, then test `AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)` separately from the NER model. The user reports having access; verify by loading rather than assuming.
3. Run `tester.py legal-3.txt` only after both NER and tokenizer paths are usable; confirm `pipeline_output.txt` is fresh and all source chunks are English.
4. Implement the agreed optional LLM entity-review stage after defining provider/local model, evidence-span schema, privacy policy, timeout/failure behavior, and deterministic validation. Keep the NER-only fallback usable.
5. Evaluate entity precision/recall and alias merging against manually annotated legal-document samples before treating the registry as authoritative.
6. Keep actual translation, consistency repair, persistence, and final assembly out of scope until separately specified.
