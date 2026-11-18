from app.rag.chunking import chunk_law_text

SAMPLE = """9/2023. (V. 25.) ÉKM rendelet
az épületek energetikai jellemzőinek meghatározásáról
2023.11.02.
A felhatalmazás szövege, amely nem kerül chunkba.
1. § (1) E rendelet hatálya az épületekre terjed ki, kivéve a kivételeket.
(2) E rendelet hatálya nem terjed ki a műemlék épületre és a helyi védelem alatt álló épületre.
2. § E rendelet alkalmazásában
1. alternatív rendszer: a megújuló energiaforrásokon alapuló decentralizált energiaellátási rendszer;
2. hőhíd: a határoló szerkezet olyan része, ahol többdimenziós hőáram alakul ki;
3. § (1) Épületet úgy kell tervezni, hogy annak energetikai jellemzői megfeleljenek.
12. §2
1. melléklet a 9/2023. (V. 25.) ÉKM rendelethez
Általános követelmények
1. A határoló és nyílászáró szerkezetek követelményei
1.1. A hőátbocsátási tényező követelményértékei
[TÁBLÁZAT]
Épülethatároló szerkezet | Követelményérték (W/m2K)
Homlokzati fal | 0,24
[/TÁBLÁZAT]
2. melléklet a 9/2023. (V. 25.) ÉKM rendelethez
3R03473.pdf
1
Lásd a közleményt.
"""


def _by_hely():
    return {c.metadata["hely"]: c for c in chunk_law_text(SAMPLE)}


def test_paragraphs_are_split_per_bekezdes():
    by = _by_hely()
    assert "műemlék" in by["1. § (2) bekezdés"].text
    assert by["1. § (1) bekezdés"].text.startswith("9/2023. (V. 25.) ÉKM rendelet, 1. § (1) bekezdés:")


def test_definitions_become_separate_chunks():
    by = _by_hely()
    assert "hőhíd:" in by["2. § 2. pont (hőhíd)"].text


def test_preamble_and_repealed_paragraph_are_dropped():
    chunks = chunk_law_text(SAMPLE)
    assert not any("felhatalmazás" in c.text for c in chunks)
    assert "12. §" not in _by_hely()


def test_annex_table_stays_with_its_point_and_heading():
    text = _by_hely()["1. melléklet 1.1. pont"].text
    assert "Homlokzati fal | 0,24" in text
    assert "követelményei" in text
    assert "(Általános követelmények)" in text


def test_short_sibling_points_are_not_merged_as_headings():
    text = """1. § (1) E rendelet hatálya az épületekre terjed ki, kivéve a kivételeket.
7. melléklet a 9/2023. (V. 25.) ÉKM rendelethez
Súlyozó tényezők
1. A súlyozott energetikai teljesítményt négyféle súlyozó tényezővel lehet megadni:
1.1. nem megújuló primerenergia átalakítási tényező: fnren
1.2. megújuló primerenergia átalakítási tényező: fren
1.3. teljes primerenergia átalakítási tényező: ftot
"""
    by = {c.metadata["hely"]: c.text for c in chunk_law_text(text)}
    assert "négyféle" in by["7. melléklet 1.1. pont"]  # a címsor a gyerekpontjához tapad
    assert "fren" in by["7. melléklet 1.2. pont"]
    assert "fren" not in by["7. melléklet 1.3. pont"]
    assert "ftot" not in by["7. melléklet 1.2. pont"]


def test_pdf_annex_gets_placeholder_and_footnotes_are_removed():
    chunks = chunk_law_text(SAMPLE)
    assert "PDF" in _by_hely()["2. melléklet"].text
    assert not any("Lásd a közleményt" in c.text for c in chunks)


def test_metadata_has_source_and_time_state():
    meta = chunk_law_text(SAMPLE)[0].metadata
    assert meta["idoallapot"] == "2023.11.02."
    assert meta["rendelet"].startswith("9/2023.")
