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

    report_lines = [
        "English Document Preprocessing Report",
        "====================================",
        f"Document ID: {chunks[0].documentId if chunks else 'doc-001'}",
        f"Chunk count: {len(chunks)}",
        f"Entity/term candidate count: {len(queue.entities)}",
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
            "Gender is only populated when an extractor supplies it; the current",
            "English NER and YAKE extractors do not infer gender, so null is expected.",
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