import re
from dataclasses import dataclass, field

from app.domain import Chunk

RENDELET = "9/2023. (V. 25.) ÉKM rendelet"
MAX_CHARS = 1800
MIN_CHARS = 30  # pl. a hatályon kívüli 12. § kiesik
HEADING_MAX_CHARS = 120  # melléklet-pont címsor, ha ennél rövidebb és nincs a végén . vagy ;

PARA_RE = re.compile(r"^(\d+)\. §\s*(.*)$")
SUBPARA_RE = re.compile(r"^\((\d+[a-z]?)\)\s+(.*)$")
DEFINITION_RE = re.compile(r"^(\d+)\.\s+([^:]{1,80}):\s+(.*)$")
ANNEX_RE = re.compile(r"^(\d+)\. melléklet\b")
POINT_RE = re.compile(r"^(\d+(?:\.\d+)*)\.(?:\s+(.*))?$")
PDF_RE = re.compile(r"^\S+\.pdf$", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.$")


@dataclass
class _Block:
    hely: str
    tipus: str
    annex: str | None = None
    context: str = ""
    lines: list[str] = field(default_factory=list)
    point: str | None = None  # melléklet-pont száma, pl. "1.2"


def _strip_footnotes(lines: list[str]) -> list[str]:
    """Lábjegyzetek a fájl végén: szám + szöveg sorpárok."""
    while len(lines) >= 2 and lines[-2].isdigit():
        lines = lines[:-2]
    return lines


def _is_heading(block: _Block) -> bool:
    text = " ".join(block.lines)
    return (
        block.annex is not None
        and block.tipus == "melleklet"
        and len(text) < HEADING_MAX_CHARS
        and not text.endswith((".", ";"))
        and not any(" | " in line for line in block.lines)
    )


def _merge_headings(blocks: list[_Block]) -> list[_Block]:
    """A címsor-pontokat hozzáfűzi a következő ponthoz, ha az az alpontjuk (1. → 1.1.).

    A testvérpont (1.2. → 1.3.) nem címsor, csak egy rövid felsorolási elem, ezért külön chunk marad.
    """
    out: list[_Block] = []
    pending: _Block | None = None
    for block in blocks:
        if pending is not None:
            is_child = (
                block.annex is not None
                and pending.annex == block.annex
                and pending.point is not None
                and block.point is not None
                and block.point.startswith(f"{pending.point}.")
            )
            if is_child:
                block.lines = pending.lines + block.lines
            else:
                out.append(pending)
            pending = None
        if _is_heading(block):
            pending = block
        else:
            out.append(block)
    if pending is not None:
        out.append(pending)
    return out


def _split_long_line(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.;])\s+", line):
        if cur and len(cur) + 1 + len(sentence) > max_chars:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        out.append(cur)
    return out


def _split_lines(lines: list[str], max_chars: int) -> list[list[str]]:
    parts: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for line in lines:
        for piece in _split_long_line(line, max_chars):
            if cur and size + len(piece) + 1 > max_chars:
                parts.append(cur)
                cur, size = [], 0
            cur.append(piece)
            size += len(piece) + 1
    if cur:
        parts.append(cur)
    return parts


def _parse_blocks(lines: list[str]) -> list[_Block]:
    blocks: list[_Block] = []
    current: _Block | None = None
    par_no: str | None = None
    annex_no: str | None = None
    annex_title = ""
    in_table = False

    for line in lines:
        if line == "[TÁBLÁZAT]":
            in_table = True
            continue
        if line == "[/TÁBLÁZAT]":
            in_table = False
            continue
        if in_table:
            if current is not None:
                current.lines.append(line)
            continue

        annex_match = ANNEX_RE.match(line)
        if annex_match:
            annex_no, annex_title, par_no, current = annex_match.group(1), "", None, None
            continue

        if annex_no is not None:
            point = POINT_RE.match(line)
            if point:
                current = _Block(
                    f"{annex_no}. melléklet {point.group(1)}. pont",
                    "melleklet",
                    annex_no,
                    annex_title,
                    [line],
                    point=point.group(1),
                )
                blocks.append(current)
            elif PDF_RE.match(line) and current is None:
                note = (
                    f"A {annex_no}. melléklet tartalma az eredeti oldalon csak PDF-ben ({line}) érhető el, "
                    "ezért ebben a tudásbázisban nem szerepel."
                )
                blocks.append(_Block(f"{annex_no}. melléklet", "melleklet_pdf", None, "", [note]))
            elif current is not None:
                current.lines.append(line)
            else:
                annex_title = f"{annex_title} {line}".strip()
            continue

        para = PARA_RE.match(line)
        if para:
            par_no, rest = para.group(1), para.group(2)
            sub = SUBPARA_RE.match(rest)
            if sub:
                current = _Block(f"{par_no}. § ({sub.group(1)}) bekezdés", "paragrafus", lines=[sub.group(2)])
            else:
                current = _Block(f"{par_no}. §", "paragrafus", lines=[rest])
            blocks.append(current)
            continue
        if par_no is None:
            continue  # bevezető

        if par_no == "2":
            definition = DEFINITION_RE.match(line)
            if definition:
                num, term, text = definition.groups()
                current = _Block(f"2. § {num}. pont ({term})", "fogalom", lines=[f"{term}: {text}"])
                blocks.append(current)
                continue

        sub = SUBPARA_RE.match(line)
        if sub:
            current = _Block(f"{par_no}. § ({sub.group(1)}) bekezdés", "paragrafus", lines=[sub.group(2)])
            blocks.append(current)
        elif current is not None:
            current.lines.append(line)

    return blocks


def chunk_law_text(text: str, rendelet: str = RENDELET) -> list[Chunk]:
    lines = _strip_footnotes([ln.strip() for ln in text.splitlines() if ln.strip()])
    idoallapot = next((ln for ln in lines[:6] if DATE_RE.match(ln)), "")

    chunks: list[Chunk] = []
    for block in _merge_headings(_parse_blocks(lines)):
        body = block.lines
        if len(" ".join(body)) < MIN_CHARS:
            continue
        parts = _split_lines(body, MAX_CHARS)
        for i, part in enumerate(parts, 1):
            hely = block.hely if len(parts) == 1 else f"{block.hely} ({i}/{len(parts)}. rész)"
            if i > 1 and block.annex is not None and body[0] not in part:
                part = [body[0]] + part  # cím minden részbe
            header = f"{rendelet}, {hely}"
            if block.context:
                header += f" ({block.context})"
            chunks.append(
                Chunk(
                    id=f"chunk-{len(chunks) + 1:03d}",
                    text=f"{header}:\n" + "\n".join(part),
                    metadata={"rendelet": rendelet, "hely": hely, "tipus": block.tipus, "idoallapot": idoallapot},
                )
            )
    return chunks
