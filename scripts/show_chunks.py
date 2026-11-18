"""Használat: uv run python -m scripts.show_chunks [fájl] [hely-részlet]"""

import sys
from pathlib import Path

from app.rag.chunking import chunk_law_text


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample/9_2023_ekm.txt")
    chunks = chunk_law_text(path.read_text(encoding="utf-8"))
    if len(sys.argv) > 2:
        for c in chunks:
            if sys.argv[2] in c.metadata["hely"]:
                print(f"\n--- {c.id} ---\n{c.text}")
        return
    sizes = sorted(len(c.text) for c in chunks)
    print(f"{len(chunks)} chunk | hossz: min {sizes[0]}, medián {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    for c in chunks:
        print(f"{c.id}  {len(c.text):5d}  {c.metadata['hely']}")


if __name__ == "__main__":
    main()
