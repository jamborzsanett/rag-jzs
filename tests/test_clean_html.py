from scripts.clean_html import clean_html

HTML = """<html><head><title>x</title></head><body>
<nav>Menü</nav>
<h1>9/2023. (V. 25.) ÉKM rendelet</h1>
<p>1. § E rendelet <a href="#">hatálya</a> kiterjed.</p>
<table><tr><th>A</th><th>B</th></tr><tr><td>Homlokzati fal</td><td>0,24</td></tr></table>
<h3>Változott rendelkezések</h3><p>süti</p>
</body></html>"""


def test_clean_html_keeps_law_text_and_drops_boilerplate():
    text = clean_html(HTML)
    assert "Menü" not in text
    assert "süti" not in text
    assert "1. § E rendelet hatálya kiterjed." in text
    assert "Homlokzati fal | 0,24" in text


def test_table_header_letters_and_row_numbers_are_removed():
    html = """<html><body><h1>9/2023. (V. 25.) ÉKM rendelet</h1>
    <table><tr><td></td><th>A</th><th>B</th></tr>
    <tr><td>1</td><td>Homlokzati fal</td><td>0,24 W/m<sup>2</sup>K</td></tr></table></body></html>"""
    text = clean_html(html)
    assert "Homlokzati fal | 0,24 W/m2K" in text
    assert "| A | B" not in text
