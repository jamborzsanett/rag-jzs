"""A Nemzeti Jogszabálytárból mentett HTML-ből tiszta szöveget készít.

Használat: uv run python scripts/clean_html.py data/raw/9_2023_ekm.html data/sample/9_2023_ekm.txt
"""

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]
TITLE_RE = re.compile(r"^\d+/\d{4}\. \(.+\) .+ rendelet$")
END_MARKER = "Változott rendelkezések"
_LETTER_RE = re.compile(r"[A-Z]")


def _cell_text(cell) -> str:
    for tag in cell.find_all(["br", "p", "div"]):
        tag.insert_after(" ")
    return " ".join(cell.get_text("").split())  # különben az m<sup>2</sup>-ből "m 2" lesz


def _tables_to_text(soup: BeautifulSoup) -> None:
    """Táblázat -> soronként "cella | cella"."""
    for table in soup.find_all("table"):
        rows = [[_cell_text(c) for c in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
        # a Jogszabálytár A, B, C fejlécsorát és a sorszám-oszlopot eldobjuk
        header = rows[0] if rows else []
        if header and header[0] == "" and any(header) and all(c == "" or _LETTER_RE.fullmatch(c) for c in header):
            rows = [r[1:] for r in rows[1:]]
        lines = [" | ".join(r) for r in rows if any(r)]
        table.replace_with(NavigableString("\n[TÁBLÁZAT]\n" + "\n".join(lines) + "\n[/TÁBLÁZAT]\n"))


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    for img in soup.find_all("img"):
        if (img.get("alt") or "").lower() == "equation":
            img.replace_with(NavigableString("[KÉPLET: lásd az eredeti szöveget]"))
        else:
            img.decompose()

    _tables_to_text(soup)

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    root = soup.body or soup
    text = root.get_text("").replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]

    start = next((i for i, line in enumerate(lines) if TITLE_RE.match(line)), None)
    if start is None:
        raise SystemExit("Nem találom a jogszabály címsorát (pl. '9/2023. (V. 25.) ÉKM rendelet').")
    end = next((i for i, line in enumerate(lines) if i > start and line.startswith(END_MARKER)), len(lines))
    return "\n".join(lines[start:end]) + "\n"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    text = clean_html(src.read_text(encoding="utf-8"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(f"{dst}: {len(text.splitlines())} sor, {len(text)} karakter")


if __name__ == "__main__":
    main()
