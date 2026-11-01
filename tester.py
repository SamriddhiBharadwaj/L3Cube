import asyncio
import argparse
from pathlib import Path

from translation_pipeline import run_preprocessing_pipeline

async def main(input_path: Path, target_language: str):
    queue = await run_preprocessing_pipeline(
        str(input_path),
        document_id="doc-001",
        max_tokens=512,
        target_language=target_language,
    )

    chunks = []
    while not queue.empty():
        chunk = await queue.get()
        chunks.append(chunk)
        queue.task_done()

    lang = queue.preprocessed.language if hasattr(queue, "preprocessed") and queue.preprocessed else None
    page_headers = queue.preprocessed.page_headers if hasattr(queue, "preprocessed") and queue.preprocessed else []
    total_sentences = sum(len(c.sentences) for c in chunks)

    # Collect document-wide warnings
    doc_warnings = []
    for c in chunks:
        for w in c.warnings:
            doc_warnings.append(f"Chunk {c.unitIndex}: {w}")
    if lang and lang.code_switch_flags:
        for f in lang.code_switch_flags:
            doc_warnings.append(f"Sentence {f.sentence_id}: {f.reason} ({f.detected_language})")

    report_lines = [
        "English Document Preprocessing Report",
        "====================================",
        f"Document ID: {chunks[0].documentId if chunks else 'doc-001'}",
        f"Detected Language: {lang.detected_language if lang else 'unknown'} (confidence: {lang.confidence if lang else 0.0:.2f})",
        f"Target Language: {target_language}",
        f"Chunk count: {len(chunks)}",
        f"Total sentence records: {total_sentences}",
        f"Entity/term candidate count: {len(queue.entities)}",
        f"Page headers removed: {len(page_headers)}",
        "",
        "WARNINGS SUMMARY",
        "================",
        "\n".join(f"- {w}" for w in doc_warnings) if doc_warnings else "No warnings flagged.",
        "",
        "Chunk fields are shown separately. Context is neighboring text and is not",
        "part of srcText; its overlap with adjacent chunks is intentional.",
        "",
        "TRANSLATION CHUNKS",
        "==================",
    ]

    for chunk in chunks:
        report_lines.extend(
            [
                "",
                f"Chunk {chunk.unitIndex}",
                f"  Chunk ID: {chunk.chunkId}",
                f"  Section ID: {chunk.sectionId}",
                f"  Paragraph ID: {chunk.paragraphId}",
                f"  Status: {chunk.status}",
                f"  Tokens: {chunk.contentTokenCount} content + {chunk.contextTokenCount} context (budget: 512)",
                f"  Unknown token rate: {chunk.unknownTokenRate:.2%}",
                f"  Warnings: {', '.join(chunk.warnings) or 'none'}",
                f"  Sentence count: {len(chunk.sentences)}",
                "  Preceding context:",
                f"    {chunk.precedingContext or '[none]'}",
                "  Source text (translation input):",
                f"    {chunk.srcText}",
                "  Following context:",
                f"    {chunk.followingContext or '[none]'}",
            ]
        )

    report_lines.extend(
        [
            "",
            "ENTITY AND TERM REGISTRY",
            "========================",
            "Types: person, place, organization, proper_noun, technical_term, case_number, citation, identifier, date, amount.",
            "Gender is inferred from honorific prefixes when available.",
        ]
    )
    for index, entity in enumerate(queue.entities, start=1):
        report_lines.extend(
            [
                "",
                f"Candidate {index}: {entity.sourceTerm}",
                f"  Entity ID: {entity.entityId}",
                f"  Type: {entity.termType}",
                f"  Target name: {entity.targetName or '[not assigned]'}",
                f"  Gender: {entity.gender or '[not provided]'}",
                f"  Mentions: {entity.mentionCount}",
                f"  First sentence ID: {entity.firstSentenceId or '[not recorded]'}",
                f"  Source: {entity.source}",
                f"  Confidence: {f'{entity.confidence:.2f}' if entity.confidence is not None else '[not provided]'}",
            ]
        )

    output_path = Path(__file__).with_name("pipeline_output.txt")
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks and {len(queue.entities)} entity candidates to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and chunk an English .txt document.")
    parser.add_argument("input_file", type=Path, help="Path to a UTF-8 English .txt document")
    parser.add_argument("--target-language", default="hin_Deva", help="IndicTrans2 target language code")
    args = parser.parse_args()
    asyncio.run(main(args.input_file, args.target_language))